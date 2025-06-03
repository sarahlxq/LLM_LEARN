#!/bin/bash
export MODEL_WEIGHTS_PATH=/app/models/finetuned-model/m3e-base
supervisord -c supervicord.conf 
tail -f m3e_app.err.log &
tail -f rerank_app.err.log 



