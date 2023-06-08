import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def array_to_sparseTensor(hitList, edepList):
    ME.clear_global_coordinate_manager()

    hitCoordTensors = []
    hitFeatureTensors = []
    
    edepCoordTensors = []
    edepFeatureTensors = []

    LarpixPadCoordTensors = []
    LarpixPadFeatureTensors = []

    EdepPadCoordTensors = []
    EdepPadFeatureTensors = []

    # GenericPadCoordTensors = []
    # GenericPadFeatureTensors = []
    
    for hits, edep in zip(hitList, edepList):
        
        # trackX, trackZ, trackY, dE = edep
        edepX = edep['x']
        edepY = edep['y']
        edepZ = edep['z']
        dE = edep['dE']
        
        edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
        edepFeature = torch.FloatTensor(np.array([dE])).T
                
        edepCoordTensors.append(edepCoords)
        edepFeatureTensors.append(edepFeature)

        # hitsX, hitsY, hitsZ, hitsQ = hits
        hitsX = hits['x']
        hitsY = hits['y']
        hitsZ = hits['z']
        hitsQ = hits['q']

        hitCoords = torch.FloatTensor(np.array([hitsX, hitsY, hitsZ])).T
        hitFeature = torch.FloatTensor(np.array([hitsQ])).T
            
        hitCoordTensors.append(hitCoords)
        hitFeatureTensors.append(hitFeature)

        LarpixPadCoords = edepCoords
        LarpixPadFeature = torch.zeros((LarpixPadCoords.shape[0], 1))
        
        LarpixPadCoordTensors.append(LarpixPadCoords)
        LarpixPadFeatureTensors.append(LarpixPadFeature)

        EdepPadCoords = hitCoords
        EdepPadFeature = torch.zeros((EdepPadCoords.shape[0], 1))
        
        EdepPadCoordTensors.append(EdepPadCoords)
        EdepPadFeatureTensors.append(EdepPadFeature)

        # GenericPadCoords = hitCoords # add to this list everything +/- 1 in each dimension
        # GenericPadFeature = torch.zeros((GenericPadCoords.shape[0], 1))

        # GenericPadCoordTensors.append(GenericPadCoords)
        # GenericPadFeatureTensors.apend(GenericPadFeature)
        
    hitCoords, hitFeature = ME.utils.sparse_collate(hitCoordTensors, 
                                                    hitFeatureTensors,
                                                    dtype = torch.int32)
                
    edepCoords, edepFeature = ME.utils.sparse_collate(edepCoordTensors, 
                                                      edepFeatureTensors,
                                                      dtype = torch.int32)
    
    LarpixPadCoords, LarpixPadFeature = ME.utils.sparse_collate(LarpixPadCoordTensors, 
                                                                LarpixPadFeatureTensors,
                                                                dtype = torch.int32)

    EdepPadCoords, EdepPadFeature = ME.utils.sparse_collate(EdepPadCoordTensors, 
                                                            EdepPadFeatureTensors,
                                                            dtype = torch.int32)

    # GenericPadCoords, GenericPadFeature = ME.utils.sparse_collate(GenericPadCoordTensors,
    #                                                               GenericPadFeatureTensors,
    #                                                               dtype = torch.int32)
                
    larpix = ME.SparseTensor(features = hitFeature.to(device),
                             coordinates = hitCoords.to(device))
    edep = ME.SparseTensor(features = edepFeature.to(device),
                           coordinates = edepCoords.to(device),
                           coordinate_manager = larpix.coordinate_manager,
                           )
    LarpixPad = ME.SparseTensor(features = LarpixPadFeature.to(device),
                                coordinate_map_key = edep.coordinate_map_key,
                                coordinate_manager = larpix.coordinate_manager,
                                )
    EdepPad = ME.SparseTensor(features = EdepPadFeature.to(device),
                              coordinate_map_key =  larpix.coordinate_map_key,
                              coordinate_manager = larpix.coordinate_manager,
                              )

    # GenericPad = ME.SparseTensor(features = GenericPadFeature.to(device),
    #                              coordinate_map_key = larpix.coordinate_map_key,
    #                              coordinate_manager = larpix.coordinate_manager)

    larpix = larpix + LarpixPad # + GenericPad
    edep = edep + EdepPad # + GenericPad
    
    return larpix, edep

def array_to_sparseTensor_class(inferenceList, evInfoList):
    ME.clear_global_coordinate_manager()

    infCoordTensors = []
    infFeatureTensors = []
    
    LABELS = [11,22,13,211,2212]
    PIDlabel = []
    
    for inference, evinfo in zip(inferenceList, evInfoList):

        # print ("this is one inference", inference)
        infX = inference['x']
        infY = inference['y']
        infZ = inference['z']
        infDE = inference['dE']
        infDE_err = inference['dE_err']

        infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
        infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

        infCoordTensors.append(infCoords)
        infFeatureTensors.append(infFeature)

        PIDlabel.append(LABELS.index(evinfo['primaryPID']))

    infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                    infFeatureTensors,
                                                    dtype = torch.int32)

    inference = ME.SparseTensor(features = infFeature.to(device),
                                coordinates = infCoords.to(device))

    PIDlabel = torch.LongTensor(PIDlabel).to(device)
    
    return inference, PIDlabel

def array_to_sparseTensor_class_gt(inferenceList, evInfoList):
    ME.clear_global_coordinate_manager()

    infCoordTensors = []
    infFeatureTensors = []
    
    LABELS = [11,22,13,211,2212]
    PIDlabel = []
    
    for inference, evinfo in zip(inferenceList, evInfoList):

        # print ("this is one inference", inference)
        infX = inference['x']
        infY = inference['y']
        infZ = inference['z']
        infDE = inference['dE']
        # infDE_err = inference['dE_err']

        infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
        infFeature = torch.FloatTensor(np.array([infDE])).T

        infCoordTensors.append(infCoords)
        infFeatureTensors.append(infFeature)

        PIDlabel.append(LABELS.index(evinfo['primaryPID']))

    infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                    infFeatureTensors,
                                                    dtype = torch.int32)

    inference = ME.SparseTensor(features = infFeature.to(device),
                                coordinates = infCoords.to(device))

    PIDlabel = torch.LongTensor(PIDlabel).to(device)
    
    return inference, PIDlabel
    
def array_to_sparseTensor_class_lndsm(hitList, evInfoList):
    ME.clear_global_coordinate_manager()

    hitCoordTensors = []
    hitFeatureTensors = []
    
    LABELS = [11,22,13,211,2212]
    PIDlabel = []
    
    for hit, evinfo in zip(hitList, evInfoList):

        # print ("this is one hit", hit)
        hitX = hit['x']
        hitY = hit['y']
        hitZ = hit['z']
        hitQ = hit['q']
        # infDE_err = hit['dE_err']

        hitCoords = torch.FloatTensor(np.array([hitX, hitY, hitZ])).T
        hitFeature = torch.FloatTensor(np.array([hitQ])).T

        hitCoordTensors.append(hitCoords)
        hitFeatureTensors.append(hitFeature)

        PIDlabel.append(LABELS.index(evinfo['primaryPID']))

    hitCoords, hitFeature = ME.utils.sparse_collate(hitCoordTensors,
                                                    hitFeatureTensors,
                                                    dtype = torch.int32)

    hitST = ME.SparseTensor(features = hitFeature.to(device),
                            coordinates = hitCoords.to(device))

    PIDlabel = torch.LongTensor(PIDlabel).to(device)

    return hitST, PIDlabel
    
class TransformFactoryClass:
    map = {'array_to_sparseTensor': array_to_sparseTensor,
           'array_to_sparseTensor_class': array_to_sparseTensor_class,
           'array_to_sparseTensor_class_gt': array_to_sparseTensor_class_gt,
           'array_to_sparseTensor_class_lndsm': array_to_sparseTensor_class_lndsm,
           }
    def __getitem__(self, req):
        if req in self.map:
            return self.map[req]
        else:
            return None
transformFactory = TransformFactoryClass()
