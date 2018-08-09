package mycrf;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Create by manlier 2018/8/7 16:55
 */
public class MyCRF {

    private SDVariable transitions;
    private SameDiff sd = SameDiff.create();
    private final int nLabel;
    private final float small;

    public MyCRF(int nLabel) {
        this.nLabel = nLabel;
        this.small = -1000;

        this.initialize();
    }

    private void initialize() {
        SDVariable drange = sd
                .sqrt(sd.scalar("a", 6)
                        .div(sd.var(Nd4j.create(new double[]{nLabel + 2, nLabel + 2})).sum(1)));
        transitions = sd.randomUniform("transitions", -1.0, 1.0, nLabel + 2, nLabel + 2);
    }


    /**
     * 计算损失
     *
     * @param xs shape of (batch_size, sequence_length, label_size)
     * @param ys shape of (batch_size, 1)
     * @return CRF 损失
     */
    public SDVariable calcLoss(SDVariable xs, SDVariable ys) {
        // Assign new id for extra added labels (START and END)
        SDVariable b_id = sd.var("b_id", Nd4j.create(new double[]{nLabel}));
        SDVariable e_id = sd.var("e_id", Nd4j.create(new double[]{nLabel + 1}));

        // TODO: 采用并行处理
        for (int i = 0; i < xs.getShape()[0]; i++) {
            SDVariable xs_i = xs.get(SDIndex.point(i));
            SDVariable ys_i = ys.get(SDIndex.point(i));

            long s_len = xs_i.getShape()[0]; // 语句含有多少单词

            SDVariable b_s = sd.var("b_s", Nd4j.zeros(nLabel + 2).add(small).putScalar(nLabel, 0));
            SDVariable e_s = sd.var("e_s", Nd4j.zeros(nLabel + 2).add(small).putScalar(nLabel + 1, 0));
            SDVariable observations = sd.concat(1, xs_i, sd.one("ones", new long[]{s_len, 2}));

            // 获得观测值
            observations = sd.concat(0, b_s, observations, e_s);

            // 根据实际标注计算真实路径分
            // Emission score of the real path
            SDVariable realPathScore = sum(xs_i, createRangeIdx((int) s_len), getIdxsFromVector(ys_i.getArr().toIntVector()));

            // Transition score of the real path
            SDVariable paddedTagsIds = sd.concat(0, b_id, ys_i, e_id);

            int[] vector = paddedTagsIds.getArr().toIntVector();

            // Score(real path) = Score(Emission score) + Score(Transition score)
            realPathScore.addi(sum(transitions,
                    getIdxsFromVector(Arrays.copyOfRange(vector, 0, (int) s_len + 1)),
                    getIdxsFromVector(Arrays.copyOfRange(vector, 1, (int) s_len + 2))));

            // 计算当前语句的所有可能路径的总分


        }
        return null;
    }

    private SDVariable forward(SDVariable observations, SDVariable transitions) {
        return null;
    }

    private List<Integer> getIdxsFromVector(int[] vector) {
        List<Integer> idxs = new LinkedList<>();
        for (int i : vector) {
            idxs.add(i);
        }
        return idxs;
    }

    private List<Integer> createRangeIdx(int end) {
        List<Integer> idxs = new LinkedList<>();
        for (int i = 0; i < end; i++) {
            idxs.add(i);
        }
        return idxs;
    }

    private SDVariable sum(SDVariable matrix, List<Integer> xIdxs, List<Integer> yIdxs) {
        SDVariable sum = sd.var(Nd4j.scalar(0f));
        for (int i = 0; i < xIdxs.size(); i++) {
            sum.add(matrix.get(SDIndex.point(xIdxs.get(i)), SDIndex.point(yIdxs.get(i))));
        }
        return sum;
    }


    public static void main(String[] args) {
        INDArray x_i = Nd4j.arange(6).reshape(3, 2);
    }
}
