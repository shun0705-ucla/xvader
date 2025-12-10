python demo.py \
    --image input/1st.png input/3rd.png \
    --intrinsics input/1st_K.npy input/3rd_K.npy \
    --checkpoint checkpoints/checkpoint.pt \
    --conf_threshold 0.5 \
    --outdir output \
    --mask_sky