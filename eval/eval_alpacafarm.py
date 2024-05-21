from alpaca_eval import evaluate as alpaca_farm_evaluate
# from alpaca_farm.auto_annotations import alpaca_leaderboard
import datasets
import openai
import json
import os
import argparse



def main(args):

    predict_out_dir = args.output_file
    if not os.path.exists(predict_out_dir):
        os.mkdir(predict_out_dir)

    path_to_outputs = args.model_results_file
    with open(path_to_outputs, 'r') as f:
        model_results=json.load(f)

    df_leaderboard, annotations = alpaca_farm_evaluate(
                model_outputs=model_results,
                reference_outputs=args.reference_outputs,
                annotators_config=args.annotators_config,
                output_path=args.output_file,
                is_return_instead_of_print=True,
                precomputed_leaderboard=None,
                is_cache_leaderboard=False,
                caching_path=os.path.join(args.output_file, "alpaca_eval_annotator_cache.json"),
            )

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    with open(os.path.join(args.output_file, "metrics.json"), "w") as fout:
        json.dump(df_leaderboard.to_dict(), fout)


if __name__ == '__main__':
    import logging
    logging.disable(30)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_results_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reference_outputs",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--annotators_config",
        type=str,
        default="alpaca_eval_gpt4",
    )
    args = parser.parse_args()
    main(args)