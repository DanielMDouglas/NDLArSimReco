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
        edepY = edep['z']
        edepZ = edep['y']
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
