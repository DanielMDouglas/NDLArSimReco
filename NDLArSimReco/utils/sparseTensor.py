import MinkowskiEngine as ME
ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class transform:
    def __init__(self, augment = True):
        self.augment = augment

class array_to_sparseTensor(transform):
    def __call__(self, hitList, edepList):
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
        
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            edepX = xFlip*edep[xKey]
            edepY = yFlip*edep[yKey]
            edepZ = edep['z']
            dE = edep['dE']

            edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
            edepFeature = torch.FloatTensor(np.array([dE])).T
                
            edepCoordTensors.append(edepCoords)
            edepFeatureTensors.append(edepFeature)

            # hitsX, hitsY, hitsZ, hitsQ = hits
            hitsX = xFlip*hits[xKey]
            hitsY = yFlip*hits[yKey]
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
                                    coordinates = LarpixPadCoords.to(device),
                                    # coordinate_map_key = edep.coordinate_map_key,
                                    coordinate_manager = larpix.coordinate_manager,
                                    )
        EdepPad = ME.SparseTensor(features = EdepPadFeature.to(device),
                                  coordinates =  EdepPadCoords.to(device),
                                  # coordinate_map_key =  larpix.coordinate_map_key,
                                  coordinate_manager = larpix.coordinate_manager,
                                )
        
        # GenericPad = ME.SparseTensor(features = GenericPadFeature.to(device),
        #                              coordinate_map_key = larpix.coordinate_map_key,
        #                              coordinate_manager = larpix.coordinate_manager)
        
        larpix = larpix + LarpixPad # + GenericPad
        # larpix = larpix + larpix # + GenericPad
        # larpix = larpix + EdepPad # + GenericPad
        edep = edep + EdepPad # + GenericPad
            
        return larpix, edep

class array_to_sparseTensor_class(transform):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        LABELS = [11,22,13,211,2212]
        # LABELS = [11,22,13,211]
        PIDlabel = []
    
        for inference, evinfo in zip(inferenceList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
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

class array_to_sparseTensor_class_homog(transform):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        LABELS = [11,22,13,211,2212]
        PIDlabel = []
    
        for inference, evinfo in zip(inferenceList, evInfoList):
            
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = np.ones_like(inference['dE_err'])

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

class array_to_sparseTensor_primE(transform):
    # def __call__(self, inferenceList, edepList, evInfoList):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        totE = []
        primE = []
    
        # for inference, edep, evinfo in zip(inferenceList, edepList, evInfoList):
        for inference, evinfo in zip(inferenceList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = inference['dE_err']

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)
            
            # totE.append(np.sum(edep['dE']))
            primE.append(evinfo['primaryE'])
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))
            
        # self.totE = torch.FloatTensor(totE).to(device)
        primE = torch.FloatTensor(primE).to(device)
    
        return inference, primE

class array_to_sparseTensor_primE_homog(transform):
    # def __call__(self, inferenceList, edepList, evInfoList):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []

        totE = []
        primE = []
    
        # for inference, edep, evinfo in zip(inferenceList, edepList, evInfoList):
        for inference, evinfo in zip(inferenceList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = np.ones_like(inference['dE_err'])

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)

            # totE.append(np.sum(edep['dE']))
            
            primE.append(evinfo['primaryE'])
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        self.inference = ME.SparseTensor(features = infFeature.to(device),
                                         coordinates = infCoords.to(device))

        # self.totE = torch.FloatTensor(totE).to(device)
            
        self.primE = torch.FloatTensor(primE).to(device)
    
        return self.inference, self.primE

class array_to_sparseTensor_dEdx(transform):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        dEdx = []
    
        for inference, evinfo in zip(inferenceList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = inference['dE_err']

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)
            
            dEdx.append(evinfo['start_dEdx'])
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))
            
        dEdx = torch.FloatTensor(dEdx).to(device)
    
        return inference, dEdx

class array_to_sparseTensor_dEdx_homog(transform):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        dEdx = []
    
        for inference, evinfo in zip(inferenceList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = np.ones_like(inference['dE_err'])

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)
            
            dEdx.append(evinfo['start_dEdx'])
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))
            
        dEdx = torch.FloatTensor(dEdx).to(device)
    
        return inference, dEdx

class array_to_sparseTensor_totE(transform):
    def __call__(self, inferenceList, edepList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        totE = []
    
        for inference, edep in zip(inferenceList, edepList):
            
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = inference['dE_err']

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)
            
            totE.append(np.sum(edep['dE']))
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))
            
        totE = torch.FloatTensor(totE).to(device)
    
        return inference, totE

class array_to_sparseTensor_totE_homog(transform):
    def __call__(self, inferenceList, edepList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        totE = []
    
        for inference, edep in zip(inferenceList, edepList):
            
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            infDE_err = np.ones_like(inference['dE'])

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE, infDE_err])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)
            
            totE.append(np.sum(edep['dE']))
            
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))
            
        totE = torch.FloatTensor(totE).to(device)
    
        return inference, totE

class array_to_sparseTensor_class_gt(transform):
    def __call__(self, inferenceList, evInfoList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        LABELS = [11,22,13,211,2212]
        PIDlabel = []

        # print ("1", [LABELS.index(i['primaryPID'][0]) for i in evInfoList])
    
        for inference, evinfo in zip(inferenceList, evInfoList):
            
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            infX = xFlip*inference[xKey]
            infY = yFlip*inference[yKey]
            infZ = inference['z']
            infDE = inference['dE']
            # infDE_err = inference['dE_err']

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infDE])).T

            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)

            PIDlabel.append(LABELS.index(evinfo['primaryPID']))
            # print ("2", PIDlabel)

        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors,
                                                        infFeatureTensors,
                                                        dtype = torch.int32)

        inference = ME.SparseTensor(features = infFeature.to(device),
                                    coordinates = infCoords.to(device))

        PIDlabel = torch.LongTensor(PIDlabel).to(device)
        # print ("3", PIDlabel)
    
        return inference, PIDlabel
    
class array_to_sparseTensor_class_lndsm (transform):
    def __call__(self, hitList, evInfoList):
        ME.clear_global_coordinate_manager()

        hitCoordTensors = []
        hitFeatureTensors = []
    
        LABELS = [11,22,13,211,2212]
        PIDlabel = []

        for hit, evinfo in zip(hitList, evInfoList):

            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            # print ("this is one hit", hit)
            hitX = xFlip*hit[xKey]
            hitY = yFlip*hit[yKey]
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

class array_to_sparseTensor_semanticSegmentation(transform):
    def __call__(self, infList, edepList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        edepCoordTensors = []
        edepFeatureTensors = []

        infPadCoordTensors = []
        infPadFeatureTensors = []

        EdepPadCoordTensors = []
        EdepPadFeatureTensors = []
    
        for inf, edep in zip(infList, edepList):
        
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            edepX = xFlip*edep[xKey]
            edepY = yFlip*edep[yKey]
            edepZ = edep['z']
            semantic_label = edep['semantic_label'] + 1

            edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
            edepFeature = torch.FloatTensor(np.array([semantic_label], dtype = np.int32)).T
                
            edepCoordTensors.append(edepCoords)
            edepFeatureTensors.append(edepFeature)

            # hitsX, hitsY, hitsZ, hitsQ = hits
            infX = xFlip*inf[xKey]
            infY = yFlip*inf[yKey]
            infZ = inf['z']
            infdE = inf['dE']
            infdE_err = inf['dE_err']

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infdE, infdE_err])).T
            
            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)

            infPadCoords = edepCoords
            infPadFeature = torch.zeros((infPadCoords.shape[0], 1))

            infPadCoordTensors.append(infPadCoords)
            infPadFeatureTensors.append(infPadFeature)

            EdepPadCoords = infCoords
            EdepPadFeature = torch.zeros((EdepPadCoords.shape[0], 1))
        
            EdepPadCoordTensors.append(EdepPadCoords)
            EdepPadFeatureTensors.append(EdepPadFeature)
        
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors, 
                                                        infFeatureTensors,
                                                        dtype = torch.int32)
                
        edepCoords, edepFeature = ME.utils.sparse_collate(edepCoordTensors, 
                                                          edepFeatureTensors,
                                                          dtype = torch.int32)
    
        infPadCoords, infPadFeature = ME.utils.sparse_collate(infPadCoordTensors, 
                                                              infPadFeatureTensors,
                                                              dtype = torch.int32)

        EdepPadCoords, EdepPadFeature = ME.utils.sparse_collate(EdepPadCoordTensors, 
                                                                EdepPadFeatureTensors,
                                                                dtype = torch.int32)
            
        inf = ME.SparseTensor(features = infFeature.to(device),
                              coordinates = infCoords.to(device))
        edep = ME.SparseTensor(features = edepFeature.to(device),
                               coordinates = edepCoords.to(device),
                               coordinate_manager = inf.coordinate_manager,
                               )
        infPad = ME.SparseTensor(features = infPadFeature.to(device),
                                 coordinates = edepCoords.to(device),
                                 coordinate_manager = inf.coordinate_manager,
        )
        EdepPad = ME.SparseTensor(features = EdepPadFeature.to(device),
                                  coordinate_map_key =  inf.coordinate_map_key,
                                  coordinate_manager = inf.coordinate_manager,
                              )
        
        inf = inf + infPad # + GenericPad
        edep = edep + EdepPad # + GenericPad
        edep.features[:] -= 1
            
        return inf, edep

class array_to_sparseTensor_semanticSegmentation_homog(transform):
    def __call__(self, infList, edepList):
        ME.clear_global_coordinate_manager()

        infCoordTensors = []
        infFeatureTensors = []
    
        edepCoordTensors = []
        edepFeatureTensors = []

        infPadCoordTensors = []
        infPadFeatureTensors = []

        EdepPadCoordTensors = []
        EdepPadFeatureTensors = []
    
        for inf, edep in zip(infList, edepList):
        
            if self.augment:
                diagFlip = np.random.choice([False, True])
                if diagFlip:
                    xKey, yKey = 'y', 'x'
                else:
                    xKey, yKey = 'x', 'y'
                xFlip = np.random.choice([-1, 1])
                yFlip = np.random.choice([-1, 1])
            else:
                xKey, yKey = 'x', 'y'
                xFlip = 1
                yFlip = 1

            edepX = xFlip*edep[xKey]
            edepY = yFlip*edep[yKey]
            edepZ = edep['z']
            semantic_label = edep['semantic_label'] + 1

            edepCoords = torch.FloatTensor(np.array([edepX, edepY, edepZ])).T
            edepFeature = torch.FloatTensor(np.array([semantic_label], dtype = np.int32)).T
                
            edepCoordTensors.append(edepCoords)
            edepFeatureTensors.append(edepFeature)

            # hitsX, hitsY, hitsZ, hitsQ = hits
            infX = xFlip*inf[xKey]
            infY = yFlip*inf[yKey]
            infZ = inf['z']
            infdE = inf['dE']
            infdE_err = np.ones_like(infdE)

            infCoords = torch.FloatTensor(np.array([infX, infY, infZ])).T
            infFeature = torch.FloatTensor(np.array([infdE, infdE_err])).T
            
            infCoordTensors.append(infCoords)
            infFeatureTensors.append(infFeature)

            infPadCoords = edepCoords
            infPadFeature = torch.zeros((infPadCoords.shape[0], 1))

            infPadCoordTensors.append(infPadCoords)
            infPadFeatureTensors.append(infPadFeature)

            EdepPadCoords = infCoords
            EdepPadFeature = torch.zeros((EdepPadCoords.shape[0], 1))
        
            EdepPadCoordTensors.append(EdepPadCoords)
            EdepPadFeatureTensors.append(EdepPadFeature)
        
        infCoords, infFeature = ME.utils.sparse_collate(infCoordTensors, 
                                                        infFeatureTensors,
                                                        dtype = torch.int32)
                
        edepCoords, edepFeature = ME.utils.sparse_collate(edepCoordTensors, 
                                                          edepFeatureTensors,
                                                          dtype = torch.int32)
    
        infPadCoords, infPadFeature = ME.utils.sparse_collate(infPadCoordTensors, 
                                                              infPadFeatureTensors,
                                                              dtype = torch.int32)

        EdepPadCoords, EdepPadFeature = ME.utils.sparse_collate(EdepPadCoordTensors, 
                                                                EdepPadFeatureTensors,
                                                                dtype = torch.int32)
            
        inf = ME.SparseTensor(features = infFeature.to(device),
                              coordinates = infCoords.to(device))
        edep = ME.SparseTensor(features = edepFeature.to(device),
                               coordinates = edepCoords.to(device),
                               coordinate_manager = inf.coordinate_manager,
                               )
        infPad = ME.SparseTensor(features = infPadFeature.to(device),
                                 coordinates = edepCoords.to(device),
                                 coordinate_manager = inf.coordinate_manager,
        )
        EdepPad = ME.SparseTensor(features = EdepPadFeature.to(device),
                                  coordinate_map_key =  inf.coordinate_map_key,
                                  coordinate_manager = inf.coordinate_manager,
                              )
        
        inf = inf + infPad # + GenericPad
        edep = edep + EdepPad # + GenericPad
        edep.features[:] -= 1
            
        return inf, edep

    
class TransformFactoryClass:
    map = {'array_to_sparseTensor': array_to_sparseTensor,
           'array_to_sparseTensor_class': array_to_sparseTensor_class,
           'array_to_sparseTensor_class_homog': array_to_sparseTensor_class_homog,
           'array_to_sparseTensor_primE': array_to_sparseTensor_primE,
           'array_to_sparseTensor_primE_homog': array_to_sparseTensor_primE_homog,
           'array_to_sparseTensor_dEdx': array_to_sparseTensor_dEdx,
           'array_to_sparseTensor_dEdx_homog': array_to_sparseTensor_dEdx_homog,
           'array_to_sparseTensor_class_gt': array_to_sparseTensor_class_gt,
           'array_to_sparseTensor_class_lndsm': array_to_sparseTensor_class_lndsm,
           'array_to_sparseTensor_totE': array_to_sparseTensor_totE,
           'array_to_sparseTensor_totE_homog': array_to_sparseTensor_totE_homog,
           'array_to_sparseTensor_semanticSegmentation': array_to_sparseTensor_semanticSegmentation,
           'array_to_sparseTensor_semanticSegmentation_homog': array_to_sparseTensor_semanticSegmentation_homog,
           }
    def __getitem__(self, req):
        if req in self.map:
            return self.map[req]
        else:
            return None
transformFactory = TransformFactoryClass()
