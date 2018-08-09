package org.manlier.tokenizer.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Create by manlier 2018/7/30 10:32
 */
public class CRFOutputLayer extends BaseOutputLayer<org.manlier.tokenizer.conf.layers.CRFOutputLayer> {

    public CRFOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public CRFOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        
    }

    @Override
    protected INDArray getLabels2d(LayerWorkspaceMgr layerWorkspaceMgr, ArrayType arrayType) {
        return null;
    }
}
