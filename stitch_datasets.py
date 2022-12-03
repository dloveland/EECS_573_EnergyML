import pandas as pd
import argparse

def main(dirname):
    if dirname == "percomb_runs" or dirname == "tuning_runs" or dirname == "bayes_runs":
        for model in ["rf","xgb"]:
            for dataset in ["bank","maternal","winequality"]:
                    pd_result = pd.read_csv(f"{dirname}/results_{model}_{dataset}.csv")
                    pd_energy = pd.read_csv(f"{dirname}/energy_{model}_{dataset}.csv")
                    pd_result["Energy (J)"] = pd_energy["Energy"].astype(float)
                    pd_result["Runtime (ms)"] = pd_energy["Runtime (ms)"].astype(float)
                    pd_result["Power (W)"] = pd_result["Energy (J)"]/(pd_result["Runtime (ms)"]*1000)
                    pd_result.to_csv(f"{dirname}/combined_results_{model}_{dataset}.csv")
    else:
        print("Invalid directory")
        return
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str,required=True)
    args = parser.parse_args()

    main(args.dirname)
