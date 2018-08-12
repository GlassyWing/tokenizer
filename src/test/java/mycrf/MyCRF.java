package mycrf;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Create by manlier 2018/8/7 16:55
 */
public class MyCRF {

    private SDVariable transitions;
    private SameDiff sd;
    private final int nLabel;
    private final float small;

    public MyCRF(int nLabel) {
        this.nLabel = nLabel;
        this.small = -1000;
        this.sd = SameDiff.create();
        this.initialize();
    }

    public MyCRF(int nLabel, SameDiff sd) {
        this.nLabel = nLabel;
        this.small = -1000;
        this.sd = sd;
        this.initialize();
    }

    /**
     * 参数初始化
     */
    private void initialize() {
        SDVariable drange = sd.sqrt(sd.scalar("a", 6)
                .div(sd.var(Nd4j.create(new double[]{nLabel + 2, nLabel + 2})).sum(1)));
        transitions = drange.mul(sd.randomUniform("transitions", -1.0, 1.0, nLabel + 2, nLabel + 2));
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

        SDVariable totalLoss = sd.scalar("total_loss", 0);

        // TODO: 采用并行处理
        for (int i = 0; i < xs.getShape()[0]; i++) {
            SDVariable xs_i = xs.get(SDIndex.point(i));   // (sequence_length, nLabel)
            SDVariable ys_i = sd.expandDims(ys.get(SDIndex.point(i)), 1);   // (sequence_length, 1)

            System.err.println(xs_i.getArr());

            sd.exec();
            long s_len = xs_i.getShape()[0]; // 语句含有多少单词

            SDVariable b_s = sd.var(Nd4j.zeros(nLabel + 2)
                    .add(small).putScalar(nLabel, 0));                            // (1, nLabel + 2)
            SDVariable e_s = sd.var(Nd4j.zeros(nLabel + 2)     // (1, nLabel + 2)
                    .add(small).putScalar(nLabel + 1, 0));
            SDVariable observations = sd.concat(1, xs_i,
                    sd.var(Nd4j.ones(s_len, 2)));                    // (sequence_length, nLabel + 2)

            // 获得观测值
            observations = sd.concat(0, b_s, observations, e_s);           // (sequence + 2, nLabel + 2)

            // 根据实际标注计算真实路径分
            // Emission score of the real path
            SDVariable realPathScore = sum(xs_i, createRangeIdx((int) s_len), getIdxsFromVector(ys_i.getArr().toIntVector()));

            // Transition score of the real path
            SDVariable paddedTagsIds = sd.concat(0, b_id, ys_i, e_id);

            sd.exec();
            int[] vector = paddedTagsIds.getArr().toIntVector();

            // Score(real path) = Score(Emission score) + Score(Transition score)
            realPathScore.addi(sum(transitions,
                    getIdxsFromVector(Arrays.copyOfRange(vector, 0, (int) s_len + 1)),
                    getIdxsFromVector(Arrays.copyOfRange(vector, 1, (int) s_len + 2))));

            // 计算当前语句的所有可能路径的总分
            SDVariable allPathsScores = forward(observations, transitions);

            // 当前句子的CRF代价
            SDVariable currCost = realPathScore.sub(allPathsScores).neg();

            totalLoss.addi(currCost);

        }
        return totalLoss;
    }

    private SDVariable computeScores(SDVariable obs, SDVariable previous, SDVariable transitions) {
        previous = sd.reshape(previous, previous.getShape()[0], 1);
        obs = sd.reshape(obs, 1, obs.getShape()[0]);
        sd.exec();
        return sd.var(previous.getArr().broadcast(nLabel + 2, nLabel + 2)
                .add(obs.getArr().broadcast(nLabel + 2, nLabel + 2)))
                .add(transitions);
    }

    private SDVariable recurrence(SDVariable obs, SDVariable previous, SDVariable transitions) {
        sd.exec();
        return sd.logSumExp(computeScores(obs, previous, transitions), 0);
    }

    private Pair<SDVariable, SDVariable> viterbi(SDVariable obs, SDVariable previous, SDVariable transitions) {
        SDVariable scores = computeScores(obs, previous, transitions);

        // 获得每一列的最大值
        SDVariable out = scores.max(0);

        // 获得每一列的最大值的列指针
        SDVariable out2 = scores.argmax(0);

        return new Pair<>(out, out2);
    }

    public SDVariable predict(SDVariable observations) {
        // 初始观测值，无意义
        SDVariable previous = observations.get(SDIndex.point(0));  // (nLabel + 2)
        // alpha_0 which stores max scores of each word.
        SDVariable alpha0 = sd.expandDims(previous, 0);       // (1, nLabel + 2)
        // alpha_1 which stores max scores idxs of each word.
        SDVariable alpha1 = sd.zerosLike(alpha0);                  // (1, nLabel + 2)

        long seqLen = observations.getShape()[0];
        for (long i = 1; i < seqLen; i++) {
            SDVariable obs = observations.get(SDIndex.point(i));   // (nLabel + 2)
            Pair<SDVariable, SDVariable> tuple = viterbi(obs, previous, transitions);

            previous = tuple.getFirst();             // (nLabel + 2)
            SDVariable maxIdx = tuple.getSecond();   // (nLabel + 2)

            alpha0 = sd.stack(0, alpha0, sd.expandDims(previous, 0));  // (+1, nLabel + 2)
            alpha1 = sd.stack(0, alpha1, sd.expandDims(maxIdx, 0));    // (+1, nLabel + 2)
        }

        // alpha0: (sequenceLength + 2, nLabel + 2)
        // alpha1: (sequenceLength + 2, nLabel + 2)

        // 获得最后一组: alpha0[-1]
        SDVariable initialBeta = sd.argmax(alpha0.get(SDIndex.point(alpha0.getShape()[0] - 1))); // (1,)
        SDVariable sequence = sd.expandDims(initialBeta, 0);                                // (1, 1)

        for (long i = alpha1.getShape()[0] - 1; i >= 0; i--) {
            initialBeta = miniFunForBestSeq(alpha1.get(SDIndex.point(i)), initialBeta.getArr().getInt(0));
            sequence = sd.concat(0, sequence, sd.expandDims(initialBeta, 0)); // (+1, 1)
        }

        // sequence: (sequenceLength + 2, 1)

        sequence = sd.reverse(sequence, 1);
        sequence = sequence.get(SDIndex.interval(1, (int) sequence.getShape()[0])); // (sequenceLength, 1)
        sequence = sd.permute(sequence, 1, 0);  // (1, sequenceLength)
        return sequence.get(SDIndex.point(0));              // (sequenceLength, )
    }

    private SDVariable forward(SDVariable observations, SDVariable transitions) {
        SDVariable previous = observations.get(SDIndex.point(0));
        long seqLen = observations.getShape()[0];
        for (long i = 1; i < seqLen; i++) {
            sd.exec();
            previous = recurrence(observations.get(SDIndex.point(i)), previous, transitions);
        }
        return sd.logSumExp(previous);
    }

    private SDVariable miniFunForBestSeq(SDVariable betai, int previous) {
        return betai.get(SDIndex.point(previous));
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
            sum.add(matrix.get(SDIndex.point(xIdxs.get(i)),
                    SDIndex.point(yIdxs.get(i))));
        }
        return sum;
    }


    public static void main(String[] args) {
        SameDiff sd = SameDiff.create();
        SDVariable aVar = sd.randomUniform(-1, 1, 2);

        System.out.println(sd.execAndEndResult());


    }
}
