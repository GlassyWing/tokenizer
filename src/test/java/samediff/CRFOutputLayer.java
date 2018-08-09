package samediff;


import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Create by manlier 2018/8/6 10:14
 */
public class CRFOutputLayer extends SameDiffOutputLayer {


    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable, SDVariable sdVariable1, Map<String, SDVariable> map) {
        return null;
    }

    @Override
    public String activationsVertexName() {
        return null;
    }

    @Override
    public void defineParameters(SDLayerParams sdLayerParams) {

    }

    @Override
    public void initializeParameters(Map<String, INDArray> map) {

    }

    @Override
    public InputType getOutputType(int i, InputType inputType) {
        return inputType;
    }
}
