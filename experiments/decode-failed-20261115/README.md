# decode优化失败

Archive of the stopped decode experiment. No meaningful end-to-end decode improvement was established; the exercised correctness checks passed.

The incremental TensorFold patch preserves the tested host token reuse, completed-sync flag cleanup, decode router finalization fusion, and AOT function lookup changes. Apply it to the production source snapshot dcb39b941c2bc763104bf71795878ee1c24b5ff5 (runtime base b3c5c27), after the project's existing TensorFold patches. The additional decode_router_topk.py module must be available on the AOT compiler's Python import path. check_decode_router_topk.py is its correctness-check source.

This branch preserves the code for reference and does not deploy it. Detailed experiment records and unfinished drafts are retained locally.
