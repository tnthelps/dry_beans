import time

from numpy import mean, std
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.io import arff
from sklearn.model_selection import KFold, cross_validate
import pandas


def main():
    data = arff.loadarff("datasets/dry_bean.arff")
    dataframe = pandas.DataFrame(data[0])
    input_data = dataframe.drop(columns="Class")
    output_data = dataframe.get("Class")
    output_data_array = output_data.array

    cv = CountVectorizer()
    cv_out = cv.fit_transform(output_data_array)
    cv_indices = cv_out.indices

    k_fold = KFold(n_splits=10, shuffle=True, random_state=1)

    print("data splits/training sets")
    for train, test in k_fold.split(data[0]):
        print('train: %s, test: %s' % (train, test))

    start_svc_loop(input_data, cv_indices, k_fold)

    # start_ann_loop(input_data, cv_indices, k_fold)


def start_svc_loop(input_data, cv_indices, k_fold):
    filepath_svc = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()) + " svc_results.csv"

    print("Logging svc data to local file: '%s'" % filepath_svc)

    with open(filepath_svc, "a", encoding='windows-1252') as file:
        file.write("type,c_input,kernel_input,gamma_input,mean_test_score,std_test_score,mean_error_rate,"
                   "std_error_rate,mean_fit_time,std_fit_time,mean_score_time,std_score_time")
        file.flush()

        # 'poly',
        svc_c = 0.2
        svc_kernel_dict = ['rbf', 'sigmoid']
        svc_gamma = 0.1
        svc_counter = 1

        max_gamma = 100000
        gamma_step = 1
        gamma_step_modifier = 4

        max_c = 200001
        c_step = 1
        c_step_modifier = 4

        for kernel in svc_kernel_dict:
            while svc_gamma < max_gamma:
                while svc_c < max_c:
                    print("Testing: " + str(svc_c) + ", " + kernel + ", " + str(svc_gamma))
                    file.write(
                        run_svc(
                            svc_c, kernel, svc_gamma,
                            input_data, cv_indices, k_fold
                        )
                    )

                    print("svc iterations: " + str(svc_counter))
                    svc_counter += 1
                    file.flush()
                    svc_c += c_step
                    c_step *= c_step_modifier
                svc_c = 2
                svc_gamma += gamma_step
                gamma_step *= gamma_step_modifier
            # exit while svc_c
            svc_gamma = 0.1

        # exit for kernel
        file.flush()

    print("SVC has finished computing")


def start_ann_loop(input_data, cv_indices, k_fold):
    filepath_ann = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()) + " ann_results.csv"

    print("Logging ann data to local file: '%s'" % filepath_ann)

    layer_structures = [[5], [10, 10], [10, 10, 10]]

    with open(filepath_ann, "a", encoding='windows-1252') as file:
        file.write("type,hidden_layer_sizes_input,activation_input,batch_size_input,solver_input,alpha_input,"
                   "max_iter_input,mean_test_score,std_test_score,mean_error_rate,std_error_rate,mean_fit_time,"
                   "std_fit_time,mean_score_time,std_score_time")
        file.flush()

        ann_hidden_layer_sizes_input = 50
        ann_alpha_input = 0.0001
        ann_max_iter_input = 10
        ann_batch_size_input = 10

        # layers not currently implemented
        ann_layer_count = 1

        ann_solver_input = ['lbfgs', 'sgd', 'adam']
        ann_activation_input = ['identity', 'logistic', 'tanh', 'relu']

        ann_counter = 1

        for activation in ann_activation_input:
            for solver in ann_solver_input:
                for layer_structure in layer_structures:
                    print("Testing: " + str(ann_hidden_layer_sizes_input) + ", " + str(ann_alpha_input) + ", "
                          + str(ann_max_iter_input) + ", " + activation + ", " + str(ann_batch_size_input)
                          + ", " + solver)

                    file.write(
                        run_ann(
                            *layer_structure,
                            ann_alpha_input, ann_max_iter_input,
                            activation,
                            ann_batch_size_input,
                            solver,
                            input_data, cv_indices, k_fold
                        )
                    )
                    print("ann iterations: " + str(ann_counter))
                    ann_counter += 1

                    file.flush()

                    ann_hidden_layer_sizes_input += 50
                    ann_alpha_input = ann_alpha_input * 10
                    ann_max_iter_input += 10
                    ann_batch_size_input += 10
                # exit while
                # reset the variables for next solver
                ann_hidden_layer_sizes_input = 50
                ann_alpha_input = 0.0001
                ann_max_iter_input = 10
                ann_batch_size_input = 10
            # exit solver loop
            # reset the variables for next activation
            ann_hidden_layer_sizes_input = 50
            ann_alpha_input = 0.0001
            ann_max_iter_input = 10
            ann_batch_size_input = 10
        # exit activation loop
        # flush the files
        file.flush()
    print("ANN has finished computing")


def run_ann(hidden_layer_sizes_input, alpha_input, max_iter_input, activation_input, batch_size_input,
            solver_input, input_data, cv_indices, k_fold):
    ann = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes_input,
        activation=activation_input,
        batch_size=batch_size_input,
        solver=solver_input,
        alpha=alpha_input,
        max_iter=max_iter_input,
    )

    scores = cross_validate(ann, input_data, cv_indices, cv=k_fold, n_jobs=-1)

    return ("\nANN,%.3f,%s,%.3f,%s,%3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
            % (
                hidden_layer_sizes_input,
                activation_input,
                batch_size_input,
                solver_input,
                alpha_input,
                max_iter_input,
                mean(scores["test_score"]),
                std(scores["test_score"]),
                (1 - mean(scores["test_score"])),
                (1 - std(scores["test_score"])),
                mean(scores["fit_time"]),
                std(scores["fit_time"]),
                mean(scores["score_time"]),
                std(scores["score_time"])
            )
            )


def run_svc(c_input, kernel_input, gamma_input, input_data, cv_indices, k_fold):
    svc = SVC(
        C=c_input,
        kernel=kernel_input,
        gamma=gamma_input
    )

    scores = cross_validate(svc, input_data, cv_indices, cv=k_fold, n_jobs=-1)

    return ("\nSVC,%.3f,%s,"
            "%.3f,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
            % (
                c_input,
                kernel_input,
                gamma_input,
                mean(scores["test_score"]),
                std(scores["test_score"]),
                (1 - mean(scores["test_score"])),
                (1 - std(scores["test_score"])),
                mean(scores["fit_time"]),
                std(scores["fit_time"]),
                mean(scores["score_time"]),
                std(scores["score_time"])
            )
            )






# run main method
main()
