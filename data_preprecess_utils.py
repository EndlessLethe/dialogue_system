'''
Author: Zeng Siwei
Date: 2020-08-27 18:28:47
LastEditors: Zeng Siwei
LastEditTime: 2020-08-27 18:29:30
Description: 
'''

import pandas as pd
import logging

def generate_negtive_sample(filepath_input, filepath_output, num_neg):
    """
    Input:
        The format is:
            A single turn dialog: "Q \t A\n"
            Or two similar sentence: "S1 \t S2\n"

    Output:
        The format is "sentence.a \t sentence.b 0(or 1) \n"

    """
    data_total = pd.read_csv(filepath_input, sep = "\t")
    logging.info("Total data size: " + str(data_total.shape[0]))

    n_pos = 0
    n_neg = 0
    with open(filepath_output, "w", encoding="utf-8") as f_out:
        for i in range(data_total.shape[0]):
                q = data_total.iat[i, 0]
                true_a = data_total.iat[i, 1]
                f_out.write(q + "\t" + true_a + "\t1\n")
                n_pos += 1

                cnt_neg = 0
                while cnt_neg < num_neg:
                    index_false_a = random.randint(0, data_total.shape[0])
                    false_a = data_total.iat[index_false_a, 1]
                    f_out.write(q + "\t" + false_a + "\t0\n")
                    n_neg += 1
            if (i+1) % 10000 == 0:
                logging.info("Finished {0} sentences.".format(i))

        logging.info("Generating Positive samples: " + str(n_pos))
        logging.info("Generating Negetive samples: " + str(n_neg))


    logging.info("output candidate file to:" + filepath_output)