#!/usr/bin/env bash

TEXT=iwslt14.tokenized.de-en


python fairseq/fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en 