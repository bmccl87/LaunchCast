import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 
import tensorflow as tf
from tensorflow import keras
from custom_loss import *
from LC_data_loader_forecast import *
from LC_model_Conv3D import build_conv3d_rotate
from LC_eval_attributes import *
from LC_parser import *
from gewitter_functions import *
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.utils import plot_model

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

def generate_output_figures(args, y_true, y_pred, save_dir, wandb):
    print('in the generate_output_figures function within LC_train_eval.py')
    cmap = plt.get_cmap('viridis')
    for i in range(50):
        fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(16,16))
        for t in range(4):
            axes[t,0].imshow(y_pred[i,t,:,:,0], cmap=cmap)
            axes[t,1].imshow(y_true[i,t,:,:,0], cmap=cmap)
            axes[t,2].imshow(y_pred[i,t,:,:,1], cmap=cmap)
            axes[t,3].imshow(y_true[i,t,:,:,1], cmap=cmap)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if i%10==0:
            print(i,y_true.shape[0])
        plt.tight_layout()
        plt.savefig('%squickout_%s.png'%(save_dir,i))

        try:
            wandb.log({'Output': wandb.Image('%squickout_%s.png'%(save_dir,i))})
        except Exception as e:
            print('no logging output figures to wandb')
            print(e)
        plt.close()

def generate_loss_function_figure(args,results,save_dir,wandb):
    loss = results['history']['loss']
    val_loss = results['history']['val_loss']

    fig = plt.figure(figsize=(10,10))
    plt.plot(loss,linewidth=3.0,color='blue',label='Training')
    plt.plot(val_loss,linewidth=3.0,color='blue',linestyle='dashed',label='Validation')
    plt.legend(fontsize=18)
    plt.grid('on')
    plt.xlabel('Epochs',fontsize=18)
    plt.ylabel('Loss: Unitless',fontsize=18)
    plt.title('Loss vs Epochs',fontsize=24)
    fsave='loss_plot.png'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    plt.savefig('%s%s'%(save_dir, fsave))
    try:
        wandb.log({'Loss': wandb.Image('%s%s'%(save_dir,fsave))})
    except Exception as e:
        print('no logging loss to wandb')
        print(e)
    plt.close()

    return min(loss),min(val_loss)

def binary_PD(args, y_true, y_pred,save_dir,wandb):
    print('generating the gewitter plot for the model outputs')
    thresh_bins = np.arange(0.0,1.05,0.05)
    thresh = np.arange(0.05,1.05,0.05)

    cc_labels_1d = np.ravel(y_true[:,:,:,:,0])
    cg_labels_1d = np.ravel(y_true[:,:,:,:,1])

    cc_pred_1d = np.ravel(y_pred[:,:,:,:,0])
    cg_pred_1d = np.ravel(y_pred[:,:,:,:,1])

    plt.figure(figsize=(8,10))
    plt.hist(cc_labels_1d,bins=thresh_bins)
    plt.grid('on')
    plt.title('IC Labels',fontsize=24)
    plt.savefig('%sic_labels_hist.png'%save_dir)
    try:
        wandb.log({'IC_Labels_Histogram': wandb.Image('%sic_labels_hist.png'%save_dir)})
    except Exception as e:
        print('no logging the Performance Diagram to wandb')
        print(e)
    plt.close()

    plt.figure(figsize=(8,10))
    plt.hist(cg_labels_1d, bins=thresh_bins)
    plt.grid('on')
    plt.title('CG Labels',fontsize=24)
    plt.savefig('%scg_labels_hist.png'%save_dir)
    try:
        wandb.log({'CG_Labels_Histogram': wandb.Image('%scg_labels_hist.png'%save_dir)})
    except Exception as e:
        print('no logging the Performance Diagram to wandb')
        print(e)
    plt.close()

    plt.figure(figsize=(8,10))
    plt.hist(cc_pred_1d,bins=thresh_bins)
    plt.grid('on')
    plt.title('IC Predictions: Max Percent: %s'%(f"{np.max(cc_pred_1d)*100:04f}"),fontsize=24)
    plt.savefig('%sic_pred_hist.png'%save_dir)
    try:
        wandb.log({'IC_Pred_Histogram': wandb.Image('%sic_pred_hist.png'%save_dir)})
    except Exception as e:
        print('no logging the Performance Diagram to wandb')
        print(e)
    plt.close()

    plt.figure(figsize=(8,10))
    plt.hist(cg_pred_1d,bins=thresh_bins)
    plt.grid('on')
    plt.title('CG Predictions: Max Percent: %s'%(f"{np.max(cg_pred_1d)*100:04f}"),fontsize=24)
    plt.savefig('%scg_pred_hist.png'%save_dir)
    try:
        wandb.log({'CG_Pred_Histogram': wandb.Image('%scg_pred_hist.png'%save_dir)})
    except Exception as e:
        print('no logging the Performance Diagram to wandb')
        print(e)
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax = make_performance_diagram_axis(ax, csi_cmap='Greys')
    colors = ['#ef8a62','#67a9cf']#cc,cg
    grey_color = ['#f7f7f7']
    markers = ['o','D']#cc,cg
    linestyles = ['solid','dashed']#cc,cg

    #statistics we need for performance diagram 
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

    # get performance diagram line by getting tp,fp and fn 
    tp.reset_state()
    fp.reset_state()
    fn.reset_state()

    cc_tps = tp(cc_labels_1d,cc_pred_1d)
    cc_fps = fp(cc_labels_1d,cc_pred_1d)
    cc_fns = fn(cc_labels_1d,cc_pred_1d)

    cg_tps = tp(cg_labels_1d,cg_pred_1d)
    cg_fps = fp(cg_labels_1d,cg_pred_1d)
    cg_fns = fn(cg_labels_1d,cg_pred_1d)
    
    # calc x,y of performance diagram 
    cc_pods = cc_tps/(cc_tps + cc_fns)
    cc_srs = cc_tps/(cc_tps + cc_fps)
    cc_csis = cc_tps/(cc_tps + cc_fns + cc_fps)

    cg_pods = cg_tps/(cg_tps + cg_fns)
    cg_srs = cg_tps/(cg_tps + cg_fps)
    cg_csis = cg_tps/(cg_tps + cg_fns + cg_fps)

    cc_precision, cc_recall, cc_thresholds = precision_recall_curve(cc_labels_1d, cc_pred_1d)
    cc_auc = auc(cc_recall, cc_precision)

    cg_precision, cg_recall, cg_threshold = precision_recall_curve(cg_labels_1d, cg_pred_1d)
    cg_auc = auc(cg_recall, cg_precision)
    cc_label = 'CC - Max CSI: %s, AUC: %s'%(f"{max(cc_csis):.2f}",f"{cc_auc:.2f}")
    ax.plot(np.asarray(cc_srs),np.asarray(cc_pods),
            color=colors[0],
            markerfacecolor=colors[0],
            marker=markers[0],
            markeredgecolor='black',
            label=cc_label,
            linewidth=3,
            linestyle=linestyles[0])

    cg_label = 'CG - Max CSI: %s, AUC: %s'%(f"{max(cg_csis):.2f}",f"{cg_auc:.2f}")
    ax.plot(np.asarray(cg_srs),np.asarray(cg_pods),
            color=colors[1],
            markerfacecolor=colors[1],
            marker=markers[1],
            markeredgecolor='black',
            label=cg_label,
            linewidth=3.0,
            linestyle=linestyles[0])
    ax.legend(fontsize=18)    
    plt.tight_layout()
    fsave = 'PD.png'
    plt.savefig('%s%s'%(save_dir,fsave))
    try:
        wandb.log({'PD': wandb.Image('%s%s'%(save_dir,fsave))})
    except Exception as e:
        print('no logging the Performance Diagram to wandb')
        print(e)
    plt.close()

    return {'cg_auc':cg_auc,'cg_max_csi':max(cg_csis),'cc_auc':cc_auc,'cc_auc':cc_auc,'max_cc_pred':np.max(cc_pred_1d)*100,'max_cg_pred':np.max(cg_pred_1d)}

def model_outputs_test_tfds(test_tfds,model):
    # x_test_norm = np.concatenate([x.numpy() for x,_ in test_tfds],axis=0)
    y_pred = model.predict(test_tfds,verbose=0)
    y_true = np.concatenate([y.numpy() for _,y in test_tfds],axis=0)
    return y_true, y_pred

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    return slurm_vars

#################################################################
def check_args(args):
    '''
    Check that the input arguments are rational
    '''
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
#################################################################
def generate_fname(args):
    # Label
    exp_str = args.exp_pre
    early_str =  '_early_stop_%s'%(args.early_stopping)
    rot_str = '_rot_%s'%args.rotation
    conv_act_str = '_conv_act_%s'%args.activation_conv
    if args.L2_reg>0:
        L2_str = '_L2_%s'%f"{args.L2_reg:.09f}"
    else:
        L2_str = '_L2_0.0_'
    BN_str = '_BN_%s'%args.batch_normalization
    SD_str = '_SD_%s'%f"{args.spatial_dropout:.03f}"
    pool_str = '_pool_%s_'%(args.pool_type)
    kernel_str = '_2dkernel_%s_'%(args.kernel_size)
    last_act_str = '_last_act_%s'%args.activation_last
    epoch_str = '_epochs_%s_'%args.epochs
    lr_str = '_lrate_%s_'%f"{args.lrate:.09f}"

    if args.noise==True:
        noise_str = '_noise_%s_'%(f"{args.noise_std:.03f}")
    else:
        noise_str='_no_noise'

    if args.rotate==True:
        rotate_str='_rotate_'
    else:
        rotate_str=''

    if args.label is None:
        label_str = ""
    else:
        label_str = "_label_%s"%args.label

    temp_fbase = exp_str+rot_str+early_str+conv_act_str+L2_str+BN_str+SD_str
    temp_fbase = temp_fbase+pool_str+kernel_str+last_act_str+epoch_str+noise_str
    temp_fbase = temp_fbase+rotate_str+lr_str+label_str
    return temp_fbase

def start_wandb(args):
    #load the slurm environment variables into dictionary variable for documenting into wandb
    slurm_dict = {}
    slurm_dict['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    slurm_dict['slurm_job_name'] = os.environ.get('SLURM_JOB_NAME')
    slurm_dict['slurm_job_account'] = os.environ.get('SLURM_JOB_ACCOUNT')
    slurm_dict['slurm_cpus_per_task'] = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_dict['slurm_nodelist'] = os.environ.get('SLURM_JOB_NODELIST')
    slurm_dict['slurm_partition'] = os.environ.get('SLURM_JOB_PARTITION')
    slurm_dict['slurm_num_nodes'] = os.environ.get('SLURM_JOB_NUM_NODES')
    slurm_dict['slurm_array_job_id'] = os.environ.get('SLURM_ARRAY_JOB_ID')
    slurm_dict['slurm_task_id'] = os.environ.get('SLURM_ARRAY_TASK_ID')

    #load the variables into dictionary for passing into wandb
    args_dict = vars(args)
    config_dict = {}
    for key in args_dict:
        config_dict[key] = args_dict[key]
    for key in slurm_dict:
        config_dict[key] = slurm_dict[key]

    print('starting wandb:',args.project)
    wandb_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/wandb/'
    if os.path.isdir(wandb_dir)==False:
        os.makedirs(wandb_dir)
    run = wandb.init(dir=wandb_dir,
                    project=args.project, 
                    name=generate_fname(args), 
                    notes=generate_fname(args), 
                    config=config_dict)
    wandb.log({'hostname': socket.gethostname()})
    wandb.run.log_code(".")
    return wandb

def execute_exp(args=None, multi_gpus=False,wandb=None):

    print('executing the experiment')
    #Check the arguments
    if args is None:
        parser = create_parser()
        args = parser.parse_args([])
    
    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    args.fbase = fbase
    results_dir = args.results_path+args.project+'/'+fbase+'/'
    args.results_path = results_dir
    args.checkpt_path = args.results_path

    if os.path.isdir(args.results_path)==False:
        os.makedirs(args.results_path)
    if os.path.isdir(args.checkpt_path)==False:
        os.makedirs(args.checkpt_path)

    # Check if output file already exists
    if args.force==False and os.path.isfile("%s/model.keras"%(results_dir))==True:
        # Results file does exist: exit
        print("Directory already exists: %s"%args.results_path)
        print("Model is already trained, ending the job")
        return
    
    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus
    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing
    if args.load_data==True:
        print('loading the data')
        train_tfds, val_tfds, test_tfds = tfds_2_model(args=args)
        print(train_tfds, val_tfds, test_tfds)

    if args.build_model:
        print('building the model')
        strategy = tf.distribute.MirroredStrategy()
        
        print('entering strategy')
        with strategy.scope():
            model = build_conv3d_rotate(args=args)
            opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
            if args.focal==False:
                losses = [tf.keras.losses.BinaryCrossentropy(from_logits=args.from_logits)]
            if args.focal==True:
                losses = [tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,
                                                                    alpha=0.25,
                                                                    gamma=2.0,
                                                                    from_logits=args.from_logits,
                                                                    label_smoothing=0.0,
                                                                    axis=-1,
                                                                    reduction='sum_over_batch_size',
                                                                    name='binary_focal_crossentropy')]
            metrics = [tf.keras.metrics.AUC(curve='PR',from_logits=True,name='AUC_PR'),
                        tf.keras.metrics.AUC(from_logits=True,name='AUC_ROC')]
            model.compile(optimizer=opt,loss=losses,metrics=metrics)
            print(model.summary())
            print('MODEL COMPILED SUCCESSFULLY')
    
    #Plot the model if the model is built
    if args.render and args.build_model:
        try:
            print('building the visual of the model architecture')
            render_fname = args.results_path+'model_plot.png'
            plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)
            wandb.log({'model architecture': wandb.Image(render_fname)})
        except Exception as e:
            print('exception occurred when generating the line by line diagram')
            print('Exception')
            print(e)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        return

    # Callbacks
    cbs = []
    if args.early_stopping:
        print('generating the early stopping callback')
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                        min_delta=args.min_delta, monitor=args.monitor)
        cbs.append(early_stopping_cb)

    #set the model checkpoints
    print('generating the checkpoint callback:')
    ckpt_fname = 'checkpoint.model.keras'
    print(ckpt_fname)
    cbs.append(tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpt_path+ckpt_fname,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=True,
                                                    save_freq='epoch'))

    print('generating the wandb metrics logger call back')
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    #train the model
    if args.train:
        print('generating the history: model.fit()')
        history = model.fit(train_tfds,
                            validation_data=val_tfds,
                            epochs=args.epochs,
                            use_multiprocessing=True, 
                            verbose=2,
                            callbacks=cbs)
        print('Done Training')

        # Generate results data
        results = {}
        results['history'] = history.history
        results['fname_base'] = fbase
        if wandb is not None:
            results['wandb_id'] = wandb.run.id
        results['args'] = args
        with open("%s/results.pkl"%(results_dir), "wb") as fp:
            pickle.dump(results, fp)
        
        # Save model
        if args.save_model:
            print('saving the model')
            model.save("%s/model.keras"%(results_dir))
            print('THE MODEL SAVED SUCCESSFULLY!!!!!')
        
        if args.save_output_labels:
            print('saving the test dataset model outputs and labels')
            y_true, y_pred = model_outputs_test_tfds(test_tfds=test_tfds,model=model)
            y_dict = {'y_true':y_true,'y_pred':y_pred}
            if args.from_logits==True:
                pickle.dump(y_dict,open('%s/labels_logit_outputs.pkl'%(results_dir),'wb'))
            else:
                pickle.dump(y_dicr,open('%s/labels_prob_outputs.pkl'%(results_dir),'wb'))
            print('THE MODEL OUTPUTS HAVE BEEN SAVED!!!!!')

        if args.make_PD:
            print('making the CSI, SR, POD figure')
            pd_results = binary_PD(args=args,y_true=y_true,y_pred=y_pred,save_dir=results_dir,wandb=wandb)
            results['pd_results'] = pd_results
            with open("%s/results.pkl"%(results_dir), "wb") as fp:
                pickle.dump(results, fp)
            print('THE PD FIGURE SAVED')
        
        if args.make_loss_fig:
            print('generating the loss function plots')
            min_loss, min_val_loss = generate_loss_function_figure(args=args,results=results,save_dir=results_dir,wandb=wandb)
            results['min_loss'] = min_loss
            results['min_val_loss'] = min_val_loss
            with open("%s/results.pkl"%(results_dir), "wb") as fp:
                pickle.dump(results, fp)
            print('THE LOSS FIGURE SAVED')
        
        if args.make_output_figs:
            print('making the output test dataset figures')
            fig_save_dir = '%s/output_figures/'%(results_dir)
            if os.path.isdir(fig_save_dir)==False:
                os.makedirs(fig_save_dir)
            generate_output_figures(args=args, y_true=y_true, y_pred=y_pred,save_dir=fig_save_dir,wandb=wandb)
            print('OUTPUT FIGURES GENERATED SUCCESSFULLY')
        
        if args.make_reliability_diagram:
            print('making the reliability diagram')
            LaunchCast_pred_hist_calc_np(save_dir=results_dir, y_true=y_true,y_pred=y_pred)
            LaunchCast_attributes_calc_np(save_dir=results_dir,y_true=y_true,y_pred=y_pred)
            LaunchCast_attributes_plot(load_dir=results_dir,wandb=wandb)
            print('RELIABILITY DIAGRAM GENERATED SUCCESSFULLY')

if __name__ == "__main__":
    extract_slurm_env()

    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    #start wandb
    wandb = start_wandb(args=args)
    
    visible_devices = tf.config.get_visible_devices('GPU') 
    print('visible_devices')
    print(visible_devices)
    n_visible_devices = len(visible_devices)
    wandb.log({'GPU_Info':visible_devices,'Num_GPUs':n_visible_devices})

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        visible_devices = tf.config.get_visible_devices('GPU') 
        n_visible_devices = len(visible_devices)
        print(n_visible_devices)
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    execute_exp(args, multi_gpus=n_visible_devices, wandb=wandb)
    
    print('ANOTHER MODEL DOWN...')
    print('END OF LC_TRAIN_EVAL.PY')