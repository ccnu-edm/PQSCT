from PQSCT import *

tf.compat.v1.disable_eager_execution()
print(tf.__version__)

model_path='bert_20_mean.h5'
savepath='./'

def evaluate_test(valid_data,valid_D,model):
    valid_model_pred = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    y_pred = mlb.transform(sigmoid_pre(mlb,valid_model_pred))
    y_true = mlb.transform(sigmoid_pre(mlb,valid_data[:, 1]))
    
    np.savetxt(savepath+"y_pred.txt", y_pred)
    np.savetxt(savepath+"y_true.txt", y_true)

    h = metrics.hamming_loss(y_true,y_pred)
    p = metrics.precision_score(y_true, y_pred, average='micro')
    r = metrics.recall_score(y_true, y_pred,average='micro')
    f1 = metrics.f1_score(y_true, y_pred,average='micro')
    print('hamming_loss',h)
    print('precision_score ',p)
    print('recall_score',r)
    print('f1_score ',f1)


def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    
if __name__ == '__main__':
    model = build_bert(nclass=num_classes,strategy='mean') 
    get_flops_params()

    model.load_weights(model_path)
    model.summary()

    mlb=get_mlb()
    # train_data=load_data(train_data_path)
    # valid_data=load_data(valid_data_path)
    test_data=load_data(test_data_path)

    # train_D = data_generator(train_data, shuffle=True)
    # valid_D = data_generator(valid_data, shuffle=False)
    test_D = data_generator(test_data, shuffle=False)

    dd=evaluate_test(test_data,test_D,model)
