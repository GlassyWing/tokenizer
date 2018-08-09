package samediff;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Create by manlier 2018/8/2 20:52
 */
public class L2NormalizeLambdaLayer extends SameDiffLambdaLayer {

    private int[] dimensions;

    /**
     *
     * @param dimensions Dimensions to calculate L2 norm over.
     *                   For DenseLayer/FeedForward input, this would be dimension 1
     *                   For RNNs, this would also be dimension 1(to normalize each time step separately)
     *                   For CNNs, this would be dimensions 1, 2 and 3
     */
    public L2NormalizeLambdaLayer(int... dimensions) {
        this.dimensions = dimensions;
    }

    private L2NormalizeLambdaLayer(){
        // Add a private no-arg constructor for use in JSON deserialization
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable) {
        // Note the 'keepdims' arg: this will keep the dimensions, so we can auto-broadcast the later division
        // For example, if input is shape [3, 4, 5, 6] and dimensions are [1, 2, 3] (i.e., for CNN activations)
        // the Norm2 has shape [3,1,1,1] - or if keepDims=false was used, it would have shape [3]

        SDVariable norm2 = sdVariable.norm2(true, dimensions);
        return sdVariable.div(norm2);
    }

    public int[] getDimensions() {
        return dimensions;
    }

    public void setDimensions(int[] dimensions) {
        this.dimensions = dimensions;
    }
}
