import pandas as pd
import numpy as np


def dat_to_df(folder: str, depth: int) -> pd.DataFrame:
    
    """Reads all the .dat files in a folder with the same parameter p 
    and saves their concatenation as a dataframes 

    Args:
        folder: Folder containing the files to load
        depth: Parameter p in the filenames

    Returns:
        The dataframe

    """
    
    cols = ["iter",
            "point",
            "energy",
            "fidelity",
            "variance",
            "corr_length",
            "const_kernel",
            "std_energies",
            "average_distances",
            "nit",
            "time_opt_bayes",
            "time_qaoa",
            "time_opt_kernel",
            "time_step"]
    angle_labels = sum([[f"gamma_{j+1}", f"beta_{j+1}"] for j in range(depth)], [])
    columns = [cols[0]] + angle_labels + cols[2:]
    df = pd.DataFrame()
    
    for file_path in folder.glob("*.dat"):
        file_name = str(file_path).split("/"[-1])
        if f"p_{depth}_" in str(file_name):
            #print(filepath)
            content = np.loadtxt(file_path)
            content = content[:-1, :]
            tmp_df = pd.DataFrame(content, columns=columns)
            df = df.append(tmp_df)
                 
    return df