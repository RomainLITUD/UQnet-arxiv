from heatmap_model.utils import *
from heatmap_model.interaction_dataset import *
import pickle
import pandas as pd
import time

def rotation_matrix(rad):
    psi = math.pi/2-rad
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])

def InferenceModel(model, filename, number, dataset, para, k=6):
    H = []
    Yp = []   
    
    nb = len(dataset)
    cut = list(range(0, nb, 400*4)) + [nb]
    
    for i in range(len(cut)-1):
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=4, shuffle=False)
        
        Hp = []
        for j in range(number):
            model.encoder.load_state_dict(torch.load(filename+'encoder'+ str(j) + '.pt'))
            model.decoder.load_state_dict(torch.load(filename+'decoder'+ str(j) + '.pt'))

            model.eval()
            Hi = []
            for k, data in enumerate(loader):
                print(i, j, end='\r')
                traj, splines, masker, lanefeature, adj, af, ar, c_mask = data
                lsp, heatmap = model(traj, splines, masker, lanefeature, adj, af, ar, c_mask)
                Hi.append(heatmap.detach().to('cpu').numpy())
            Hi = np.concatenate(Hi,0)
            Hp.append(Hi)
        Hp = np.stack(Hp, 0)
        Ht = np.transpose(Hp, (2,3,0,1))
        Ht = (Ht/np.sum(Ht, axis=(0,1))).transpose((2,3,0,1))
        hm = np.mean(Ht, 0)
        #sample outputs
        yp = ModalSampling(hm, para, r=3, k=6)
        Yp.append(yp)
        H.append(hm)

    Yp = np.concatenate(Yp, 0)
    H = np.concatenate(H, 0)    
    
    return Yp, H
        
def Generate_csv(trajmodel, filename, Yp):
    print('loading model and data...')
    F = Yp.reshape(-1, 2)
    testset = InferenceTraj(F)
    loader = DataLoader(testset, batch_size=16, shuffle=False)
    print(len(testset))
    data = np.load('./interaction_merge/test.npz', allow_pickle=True)
    translate = data['origin']
    R = data['radian']
    nb = len(R)
    
    frame_ = np.arange(11,41)
    
    rotate = np.array([rotation_matrix(theta) for theta in R])
    
    with open('./interaction_merge/testfile.pickle', 'rb') as f:
        testfile = pickle.load(f)
    
    with open('./interaction_merge/test_index.pickle', 'rb') as f:
        Dnew = pickle.load(f)
    samplelist = Dnew[0]
    tracklist = Dnew[1]
    
    file_id = [int(case[:-6])-1 for case in samplelist]
    case_id = [int(case[-6:]) for case in samplelist]
    track_id = [int(track) for track in tracklist]
        
    trajmodel.load_state_dict(torch.load(filename+'traj.pt'))
    
    print('Completing trajectories...')
    T = []
    for k, x in enumerate(loader):
        print(16*k, end='\r')
        traj = trajmodel(x)
        T.append(traj.detach().to('cpu').numpy())
    T = np.concatenate(T, 0).reshape(-1, 6, 29, 2)   
    T = np.concatenate([T, np.expand_dims(Yp, 2)], -2) #(N, 6, 30, 2)
    
    T = np.einsum('bknf,bfc->bknc', T, rotate)
    T = np.transpose(T, (1,2,0,3))
    T = np.transpose((T+translate), (2,0,1,3))
    print(T.shape)
    print('generating submission logs...')
    
    for i in range(17):
        print(i,'th file...', end='\r')
        D = {}
        indices = [pos for pos in range(nb) if file_id[pos]==i]
        case = np.array([case_id[index] for index in indices])
        track = np.array([track_id[index] for index in indices])
        traj = T[indices]
        
        nb_case = len(indices)
        case = list(np.repeat(case, 30))
        track = list(np.repeat(track, 30))
        #print(indices)
        
        frame = np.tile(frame_, nb_case)
        
        D['case_id'] = case
        D['track_id'] = track
        D['frame_id'] = frame
        D['timestamp_ms'] = (100*frame).tolist()
        for k in range(1,7):
            D['x'+str(k)] = traj[:,k-1,:,0].flatten().tolist()
            D['y'+str(k)] = traj[:,k-1,:,1].flatten().tolist()
            
        df = pd.DataFrame(D)
        df.sort_values(by=['case_id'])
        
        subfile = './submission/'+testfile[i][:-7]+'sub.csv'
        
        df.to_csv(subfile,index=False)