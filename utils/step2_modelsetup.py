from __future__ import print_function # pdb.set_trace() # !import code; code.interact(local=vars())
if True:
    import os,sys,datetime
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    # 
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {sys.argv}\n    Start_Time:{START_TIME}\n===========")

print('############ Printing Config Params ############')
if True:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-Archi_vrs')
    parser.add_argument('-Enh_mode')
    # 
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Enh_mode, args.Archi_vrs)
    # 
    pp = pprint.PrettyPrinter(indent=4)
    global_st = datetime.datetime.now()
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    print('\n\nUSING CPU !!!\n\n')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.enable_eager_execution()
print('############ End of Config Params ##############')
if True : ## versioning
    functions_dict = versioning(conf_DNN.Archi_vrs)
    build_denoise_model = functions_dict['build_denoise_model']
    build_dereverb_model = functions_dict['build_dereverb_model']

if 1 : ## Building Model : Default mtd.
    kwargs_model_setup_params={
        'mdl_name':conf_DNN.mdl_name,
        '_regularizer':l1_l2(l1=conf_DNN.reg_l1, l2=conf_DNN.reg_l2),
        'output_shape':conf_DNN.output_shape,
        'input_shape':conf_DNN.input_shape,
    }
    if   conf_DNN.Enh_mode=='denoise' : 
        model_denoise = build_denoise_model(**kwargs_model_setup_params)
        save_model(model_denoise, 
            model_path=conf_DNN.Enhance_model_path, 
            weights_path=conf_DNN.Enhance_weights_path)
    elif conf_DNN.Enh_mode=='dereverb' : 
        model_dereverb = build_dereverb_model(**kwargs_model_setup_params)
        save_model(model_dereverb, 
            model_path=conf_DNN.Enhance_model_path, 
            weights_path=conf_DNN.Enhance_weights_path)
    else:
        raise Exception('Error : conf_DNN.Enh_mode={}, but should either be "denoise" or "dereverb"'.format(conf_DNN.Enh_mode))


#################################################################
END_TIME=datetime.datetime.now()
print(f"===========\
Done python {sys.argv}\
    Start_Time  :{START_TIME}\
    End_Time    :{END_TIME}\
    Duration    :{END_TIME-START_TIME}\
===========")

"""
!import code; code.interact(local=vars())
"""
