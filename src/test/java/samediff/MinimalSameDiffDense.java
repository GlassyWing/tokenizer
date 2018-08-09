package samediff;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Create by manlier 2018/8/1 9:22
 */
public class MinimalSameDiffDense extends SameDiffLayer {

    private int nIn;
    private int nOut;
    private Activation activation;

    public MinimalSameDiffDense(int nIn, int nOut, Activation activation, WeightInit weightInit) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    public MinimalSameDiffDense() {

    }


    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);
        SDVariable mmul = sameDiff.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        return activation.asSameDiff("out", sameDiff, z);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, nIn, nOut);
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 1, nOut);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
        initWeights(nIn, nOut, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    public int getnIn() {
        return nIn;
    }

    public void setnIn(int nIn) {
        this.nIn = nIn;
    }

    public int getnOut() {
        return nOut;
    }

    public void setnOut(int nOut) {
        this.nOut = nOut;
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }
}
