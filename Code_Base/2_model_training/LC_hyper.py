import os
import copy
import tensorflow as tf
import keras_tuner as kt
from LC_model_64216 import build_model  # path/module name as appropriate
from LC_parser import create_parser
from LC_data_loader_forecast import norm_2_tfds_hyper,tfds_2_model

# --- (optional) small helper to clone argparse Namespace cleanly ---
def clone_args(args):
    # works for argparse.Namespace and similar objects
    return copy.deepcopy(args)

def build_model_from_hp(hp, base_args):
    args = clone_args(base_args)

    # 1) Dropout search space
    args.spatial_dropout = hp.Choice(
        "spatial_dropout",
        values=[0.0,0.05,0.10],
        default=getattr(base_args, "spatial_dropout", 0.0),
    )

    # 2) L2 regularization (log scale is usually best)
    args.L2_reg = hp.Choice("L2_reg",
    values=[0.0],
    default=getattr(base_args,"L2_reg",0.0))

    # 3) Conv activation choice (must match your act_function options)
    args.activation_conv = hp.Choice(
        "activation_conv",
        values=["relu", "leaky_relu", "gelu", "elu","tanh"],
        default=getattr(base_args, "activation_conv", "relu"),
    )

    loss_name = hp.Choice(
        "loss",
        values=["binary_crossentropy", "binary_focal_crossentropy"],
        default="binary_crossentropy",
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(args=args)
        if loss_name=="binary_crossentropy":
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif loss_name=="binary_focal_crossentropy":
            loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
        # Pick loss/metrics that match your training script.
        # Example for binary prediction logits/probs (adjust to your setup):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False),
            loss=loss,  # change if needed
            metrics=[tf.keras.metrics.AUC(curve="ROC", name="auc",from_logits=True),tf.keras.metrics.AUC(curve="PR", name="auc_pr",from_logits=True)])
    return model

def run_hyperband(train_ds, val_ds, base_args, project_dir="kt_hyperband"):
    epochs=50
    factor=3
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model_from_hp(hp, base_args),
        objective=kt.Objective("val_auc_pr", direction="max"),  # pick what you care about
        max_epochs=epochs,          # Hyperband upper cap
        factor=factor,               # downsampling factor per bracket
        hyperband_iterations=1, # repeat to reduce randomness
        directory=project_dir,
        project_name="LChyper3_%s_epochs_%s_factor_binary_focal_ce_4_5_5"%(epochs,factor),
        overwrite=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )


    best_hps = tuner.get_best_hyperparameters(num_trials=10)
    for i, hp in enumerate(best_hps):
        print(f"\nRank {i+1}")
        for k, v in hp.values.items():
            print(k, v)
        print()
        print()


def main():
    print('running the hyperband tuner for launchcast')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')

    project_dir = '/scratch/bmac87/LC_hyper/'
    if os.path.isdir(project_dir)==False:
        os.makedirs(project_dir)

    print('loading the data for the hyperband search')
    # norm_2_tfds_hyper(args=args)
    train_tfds, val_tfds, test_tfds = tfds_2_model(args=args,is_hyper=True)
    del test_tfds
    
    def cast_to_float32(X, y):
        X = {k: tf.cast(v, tf.float32) for k, v in X.items()}
        y = tf.cast(y, tf.float32)
        return X, y
    train_tfds = train_tfds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    val_tfds = val_tfds.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    
    print(train_tfds)
    print(val_tfds)
    print('data loaded for hyper and cast to float32 successfully')

    run_hyperband(train_ds=train_tfds, val_ds=val_tfds, base_args=args, project_dir=project_dir)

if __name__=='__main__':
    main()