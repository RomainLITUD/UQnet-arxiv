paralist = dict(epochs = 6, 
                batch_size = 16,
                xmax = 23,#23,
                ymin = -12,#-12,
                ymax = 75,
                resolution = 1.,
                nb_map_vectors = 5,
                nb_traj_vectors = 9,
                map_dim = 5,
                traj_dim = 8,
                nb_map_gnn = 5,
                nb_traj_gnn = 5, 
                nb_mlp_layers = 3,
                c_out_half = 32,
                c_mlp = 64,
                c_out = 96,
                encoder_nb_heads = 3,
                encoder_attention_size = 64,
                encoder_agg_mode = "cat",
                decoder_attention_size = 64,
                decoder_nb_heads = 3,
                decoder_agg_mode = "cat",
                decoder_masker = False,
                sigmax = 0.6,
                sigmay = 0.6,
                r_list = [2,4,8,16],
                kf = 1,
                model = 'densetnt',
                sample_range=1,
                use_masker=False, 
                lane2agent='lanegcn')