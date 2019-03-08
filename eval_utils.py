import numpy as np

def get_best_thresh(data,model,save_path,verbose=False):

    F_list1 = []
    thresh_list1 = np.arange(0,1,0.1)

    for thresh in thresh_list1:
        F, prec, rec = model.compute_eval_metrics_pred(data.valid,threshold=thresh,save_path=save_path)
        F_list1 += [F]

    max_value1 = max(F_list1)
    max_index1 = F_list1.index(max_value1)
    max_thresh1 = thresh_list1[max_index1]

    F_list2 = []
    thresh_list2 = np.arange(max(0,max_thresh1-0.09),min(1,max_thresh1+0.095),0.01)
    for thresh in thresh_list2:
        F, prec, rec = model.compute_eval_metrics_pred(data.valid,threshold=thresh,save_path=save_path)
        F_list2 += [F]

    max_value2 = max(F_list2)
    max_index2 = F_list2.index(max_value2)
    max_thresh2 = thresh_list2[max_index2]

    if verbose:
        model.print_params()
        print "Best F0 : "+str(max_value2)
        print "Best thresh : "+str(max_thresh2)

    return max_thresh2, max_value2

def get_best_eval_metrics(data,model,save_path,chunks=None,verbose=False):


    thresh,_ = get_best_thresh(data,model,save_path,verbose)

    F, prec, rec = model.compute_eval_metrics_pred(data.test,threshold=thresh,save_path=save_path)

    if verbose :
        print "F : "+str(F)+", pre : "+str(prec)+", rec : "+str(rec)

    return F, prec, rec
