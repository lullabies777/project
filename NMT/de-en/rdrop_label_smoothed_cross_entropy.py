# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion('rdrop_label_smoothed_cross_entropy')
class RdropLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
    
    ### Modified
    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True, k = 2):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        list_d = list(torch.split(net_prob, net_prob.size(0)//k, dim=0))
        list_d_tec = list(torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0))
        
        loss = 0.
        for i in range(k - 1):
            for j in range(i + 1, k):
                p, q = list_d[i], list_d[j]
                p_tec, q_tec = list_d_tec[i], list_d_tec[j]
                p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
                q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        
                if pad_mask is not None:
                    p_loss.masked_fill_(pad_mask, 0.)
                    q_loss.masked_fill_(pad_mask, 0.)

                if reduce:
                    p_loss = p_loss.sum()
                    q_loss = q_loss.sum()

                loss = p_loss + q_loss
        
        return loss / (k * (k - 1))
    
    def forward_rdrop(self, model, sample, optimizer, alpha, ignore_grad, k = 2,reduce=True):
        
        sample_input = sample['net_input']
        for i in range(k):
            if i == 0:
                src_tokens = sample_input['src_tokens']
                src_lengths = sample_input['src_lengths']
                prev_output_tokens = sample_input['prev_output_tokens']
            else:
                src_tokens = torch.cat([src_tokens, sample_input['src_tokens'].clone()], dim = 0)
                src_lengths = torch.cat([src_lengths, sample_input['src_lengths'].clone()], dim = 0)
                prev_output_tokens = torch.cat([prev_output_tokens, sample_input['prev_output_tokens'].clone()], dim = 0)
        
        sample_concat_input = {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        }    
        
        net_output = model(**sample_concat_input)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        temp = target.clone()
        for i in range(1, k):
            target = torch.cat([target, temp], dim=0)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        
        kl_loss = self.compute_kl_loss(model, net_output, pad_mask, k = k)
        loss += alpha * kl_loss

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        ntokens = sample['ntokens']
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output