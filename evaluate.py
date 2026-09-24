"""Complete-evidence recall@k: the share of queries whose gold chunks are ALL in the top k retrieved.

--queries is the JSONL file given to retrieve.py, with the gold chunk ids of each query as "gold_ids".
--results is the file retrieve.py wrote for it. Queries without gold ids are skipped.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    ids = [str(q.get("id", row)) for row, q in enumerate(queries)]
    if len(set(ids)) != len(ids):
        raise SystemExit("Query ids must be unique.")
    gold = {query_id: {str(i) for i in q["gold_ids"]} for query_id, q in zip(ids, queries) if q.get("gold_ids")}
    with open(args.results) as f:
        ranked = {str(r["id"]): [str(i) for i in r["chunk_ids"]] for r in map(json.loads, f)}
    if not gold:
        raise SystemExit("No query has gold_ids.")
    missing = gold.keys() - ranked.keys()
    if missing:
        raise SystemExit(f"{len(missing)} queries have no results, e.g. id {next(iter(missing))}.")
    for k in args.k:
        if any(len(ranked[query_id]) < k for query_id in gold):
            print(f"complete-evidence recall@{k}: skipped, some queries have fewer than {k} results "
                  f"(the corpus has fewer than {k} chunks, or retrieve.py ran with a smaller --top_k)")
            continue
        hits = sum(gold[query_id] <= set(ranked[query_id][:k]) for query_id in gold)
        print(f"complete-evidence recall@{k}: {100 * hits / len(gold):.2f}  ({len(gold)} queries)")


if __name__ == "__main__":
    main()
