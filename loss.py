import torch

def custom_loss(y_pred, y_true):
    '''
    cutmoized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''

    px_truth = torch.flatten(y_true[:, 0])
    py_truth = torch.flatten(y_true[:, 1])
    px_pred = torch.flatten(y_pred[:, 0])
    py_pred = torch.flatten(y_pred[:, 1])

    pt_truth = torch.sqrt(px_truth*px_truth + py_truth*py_truth)

    #px_truth1 = px_truth / pt_truth
    #py_truth1 = py_truth / pt_truth

    # using absolute response
    # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
    upar_pred = torch.sqrt(px_pred * px_pred + py_pred * py_pred) - pt_truth
    pt_cut = pt_truth > 0.
    upar_pred = upar_pred[pt_cut]
    pt_truth_filtered = pt_truth[pt_cut]

    #filter_bin0 = pt_truth_filtered < 50.
    filter_bin0 = torch.logical_and(pt_truth_filtered > 50.,  pt_truth_filtered < 100.)
    filter_bin1 = torch.logical_and(pt_truth_filtered > 100., pt_truth_filtered < 200.)
    filter_bin2 = torch.logical_and(pt_truth_filtered > 200., pt_truth_filtered < 300.)
    filter_bin3 = torch.logical_and(pt_truth_filtered > 300., pt_truth_filtered < 400.)
    filter_bin4 = pt_truth_filtered > 400.

    upar_pred_pos_bin0 = upar_pred[torch.logical_and(filter_bin0, upar_pred > 0.)]
    upar_pred_neg_bin0 = upar_pred[torch.logical_and(filter_bin0, upar_pred < 0.)]
    upar_pred_pos_bin1 = upar_pred[torch.logical_and(filter_bin1, upar_pred > 0.)]
    upar_pred_neg_bin1 = upar_pred[torch.logical_and(filter_bin1, upar_pred < 0.)]
    upar_pred_pos_bin2 = upar_pred[torch.logical_and(filter_bin2, upar_pred > 0.)]
    upar_pred_neg_bin2 = upar_pred[torch.logical_and(filter_bin2, upar_pred < 0.)]
    upar_pred_pos_bin3 = upar_pred[torch.logical_and(filter_bin3, upar_pred > 0.)]
    upar_pred_neg_bin3 = upar_pred[torch.logical_and(filter_bin3, upar_pred < 0.)]
    upar_pred_pos_bin4 = upar_pred[torch.logical_and(filter_bin4, upar_pred > 0.)]
    upar_pred_neg_bin4 = upar_pred[torch.logical_and(filter_bin4, upar_pred < 0.)]
    #upar_pred_pos_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred > 0.))
    #upar_pred_neg_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred < 0.))
    norm = torch.sum(pt_truth_filtered,dim=None)
    dev = torch.abs(torch.sum(upar_pred_pos_bin0,dim=None) + torch.sum(upar_pred_neg_bin0,dim=None))
    dev += torch.abs(torch.sum(upar_pred_pos_bin1,dim=None) + torch.sum(upar_pred_neg_bin1,dim=None))
    dev += torch.abs(torch.sum(upar_pred_pos_bin2,dim=None) + torch.sum(upar_pred_neg_bin2,dim=None))
    dev += torch.abs(torch.sum(upar_pred_pos_bin3,dim=None) + torch.sum(upar_pred_neg_bin3,dim=None))
    dev += torch.abs(torch.sum(upar_pred_pos_bin4,dim=None) + torch.sum(upar_pred_neg_bin4,dim=None))
    #dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin5) + tf.reduce_sum(upar_pred_neg_bin5))
    dev /= norm

    loss = 0.5*torch.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)

    #loss += 200.*dev
    loss += 5000.*dev
    return loss
