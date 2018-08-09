package org.manlier.tokenizer.conf.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collection;
import java.util.Map;

/**
 * Create by manlier 2018/7/30 10:10
 */
public class CRFOutputLayer extends BaseOutputLayer {

    private CRFOutputLayer(Builder builder) {
        super(builder);
    }


    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> collection,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("CRFOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());
        org.manlier.tokenizer.layers.CRFOutputLayer net = new org.manlier.tokenizer.layers.CRFOutputLayer(conf);
        net.setListeners(collection);
        net.setIndex(layerIndex);
        net.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        net.setParamTable(paramTable);
        return net;
    }

    @Override
    public ParamInitializer initializer() {
        return null;
    }

    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder() {
        }

        public Builder(LossFunctions.LossFunction lossFunction) {
            lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        @Override
        public <E extends org.deeplearning4j.nn.conf.layers.Layer> E build() {
            return null;
        }
    }
}
