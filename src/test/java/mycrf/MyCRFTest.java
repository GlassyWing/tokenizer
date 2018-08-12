package mycrf;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MyCRFTest {

    private SameDiff sd;
    private SDVariable xs;
    private SDVariable ys;
    private int nLabel;
    private int seqLen;

    @Before
    public void setUp() {


        nLabel = 2;
        seqLen = 2;
        sd = SameDiff.create();

        SDVariable a = sd.randomUniform(-1, 1, nLabel);
        SDVariable b = sd.randomUniform(-1, 1, nLabel);
        sd.exec();

        SDVariable x1 = sd.stack(0, b, a);
        SDVariable x2 = sd.stack(0, a, sd.zerosLike(a));

        xs = sd.stack(0, x1, x2);



        ys = sd.var(Transforms.floor(Nd4j.rand(new int[]{(int) xs.getShape()[0], seqLen}).mul(nLabel)));

    }


    @Test
    public void test() {

        new MyCRF(nLabel, sd).calcLoss(xs, ys);
        System.out.println(sd.execAndEndResult());
    }
}
