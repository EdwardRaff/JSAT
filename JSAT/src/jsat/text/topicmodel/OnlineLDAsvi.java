package jsat.text.topicmodel;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.math.FastMath;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class provides an implementation of <i>Latent Dirichlet Allocation</i>
 * for learning a topic model from a set of documents. This implementation is 
 * based on Stochastic Variational Inference and is meant for large collections 
 * (more than 100,000 data points) and can learn in an online fashion. <br>
 * <br>
 * For LDA it is common to set {@link #setAlpha(double) &alpha;} = 
 * {@link #setEta(double) &eta;} = 1/K, where K is the number of topics to be 
 * learned. Note that &eta; is not a learning rate parameter, as the symbol is 
 * usually used. <br>
 * 
 * For this algorithm, some potential parameter combinations (by column) for 
 * {@link #setMiniBatchSize(int) batch size}, {@link #setKappa(double) &kappa;},
 * and {@link #setTau0(double) &tau;<sub>0</sub>} are:<br>
 * <table>
 * <caption></caption>
 * <tr>
 *   <td>batch size</td>
 *   <td>256</td> 
 *   <td>1024</td>
 *   <td>4096</td>
 * </tr>
 * <tr>
 *   <td>&kappa;</td>
 *   <td>0.6</td> 
 *   <td>0.5</td>
 *   <td>0.5</td>
 * </tr>
 * <tr>
 *   <td>&tau;<sub>0</sub></td>
 *   <td>1024</td> 
 *   <td>256</td>
 *   <td>64</td>
 * </tr>
 *
 * </table><br>
 * For smaller corpuses, reducing &tau;<sub>0</sub> can improve the performance (even down to &tau;<sub>0</sub> = 1)
 * <br>
 * See:<br>
 * <ul>
 * <li>Blei, D. M., Ng, A. Y.,&amp;Jordan, M. I. (2003). <i>Latent Dirichlet 
 * Allocation</i>. Journal of Machine Learning Research, 3(4-5), 993–1022.
 * doi:10.1162/jmlr.2003.3.4-5.993</li>
 * <li>Hoffman, M., Blei, D.,&amp;Bach, F. (2010). <i>Online Learning for Latent 
 * Dirichlet Allocation</i>. In Advances in Neural Information Processing 
 * Systems (pp. 856–864). Retrieved from 
 * <a href="http://videolectures.net/site/normal_dl/tag=83534/nips2010_1291.pdf">
 * here</a></li>
 * <li>Hoffman, M. D., Blei, D. M., Wang, C.,&amp;Paisley, J. (2013). 
 * <i>Stochastic Variational Inference</i>. The Journal of Machine Learning 
 * Research, 14(1), 1303–1347.</li>
 * <li>Hoffman, M. D. (2013). <i>Lazy updates for online LDA</i>. Retrieved from
 * <a href="https://groups.yahoo.com/neo/groups/vowpal_wabbit/conversations/topics/250">here</a></li>
 * </ul>
 * @author Edward Raff
 */
public class OnlineLDAsvi implements Parameterized
{
    private double alpha = 1;
    private double eta = 1;
    private double tau0 = 128;
    private double kappa = 0.7;
    private int epochs = 1;
    private int D = -1;
    private int K = -1;
    private int W = -1;
    private int miniBatchSize = 256;
    private int t;

    /**
     * Creates a new Online LDA learner. The number of 
     * {@link #setK(int) topics}, expected number of 
     * {@link #setD(int) documents}, and the {@link #setVocabSize(int) vocabulary size}
     * must be set before it can be used. 
     */
    public OnlineLDAsvi()
    {
        K = D = W = -1;
    }
    
    /**
     * Creates a new Online LDA learner that is ready for online updates
     * @param K the number of topics to learn
     * @param D the expected number of documents to see
     * @param W the vocabulary size
     */
    public OnlineLDAsvi(int K, int D ,int W)
    {
        setK(K);
        setD(D);
        setVocabSize(W);
    }
    
    
    
    /*
     * Using lists instead of matricies b/c we want to use HDP-OnlineLDAsvi for 
     * when k is not specified <br>
     * One row for each K topics, each vec is |W| long
     * <br><br>
     * Lambda is also used to determine if the rest of the structures need to be
     * re-intialized. When lambda is {@code null} the structures need to be 
     * reinitialized. 
     */
    private List<Vec> lambda;
    private List<Lock> lambdaLocks;
    /**
     * Used to store the sum of each vector in {@link #lambda}. Updated live to avoid uncessary changes
     */
    private DoubleList lambdaSums;
    private int[] lastUsed;
    private List<Vec> ELogBeta;//See equation 6 in 2010 paper
    private List<Vec> ExpELogBeta;//See line 7 update in 2013 paper / equation (5) in 2010 paper

    /**
     * Holds the temp vector used to store gamma 
     * 
     * Gamma contains the per document update counterpart to {@link #lambda}. 
     * 
     * We need one gamma for each document, and each will have a value for all K
     * topics. 
     */
    private ThreadLocal<Vec> gammaLocal;
    /**
     * Holds the temp vector used to store the expectation of {@link #gammaLocal}
     */
    private ThreadLocal<Vec> logThetaLocal;
    /**
     * Holds the temp vector used to store the exponentiated expectation of 
     * {@link #logThetaLocal} 
     */
    private ThreadLocal<Vec> expLogThetaLocal;

    /**
     * Sets the number of topics that LDA will try to learn
     * @param K the number of topics to learn
     */
    public void setK(final int K)
    {
        if(K < 2)
            throw new IllegalArgumentException("At least 2 topics must be learned");
        this.K = K;
        gammaLocal = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(K);
            }
        };
        logThetaLocal  = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(K);
            }
        };
        expLogThetaLocal = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(K);
            }
        };
        
        lambda = null;
    }

    /**
     * Returns the number of topics to learn, or {@code -1} if <i>this</i> 
     * object is not ready to learn
     * @return the number of topics that will be learned
     */
    public int getK()
    {
        return K;
    }

    /**
     * Sets the approximate number of documents that will be observed
     * @param D the number of documents that will be observed
     */
    public void setD(int D)
    {
        if(D < 1)
            throw new IllegalArgumentException("The number of documents must be positive, not " + D);
        this.D = D;
    }

    /**
     * Returns the approximate number of documents that will be observed, or 
     * {@code -1} if <i>this</i> object is not ready to learn
     * @return the number of documents that will be observed
     */
    public int getD()
    {
        return D;
    }

    /**
     * Sets the vocabulary size for LDA, which is the number of dimensions in 
     * the input feature vectors. 
     * 
     * @param W the vocabulary size for LDA
     */
    public void setVocabSize(int W)
    {
        if(W < 1)
            throw new IllegalArgumentException("Vocabulary size must be positive, not " + W);
        this.W = W;
    }

    /**
     * Returns the size of the vocabulary for LDA, or {@code -1} if <i>this</i>
     * object is not ready to learn
     * @return the vocabulary size for LDA
     */
    public int getVocabSize()
    {
        return W;
    }
    
    /**
     * Sets the prior for the on weight vector theta. 1/{@link #setK(int) K} is 
     * a common choice. 
     * @param alpha the positive prior value
     */
    public void setAlpha(double alpha)
    {
        if(alpha <= 0 || Double.isInfinite(alpha) || Double.isNaN(alpha))
            throw new IllegalArgumentException("Alpha must be a positive constant, not " + alpha);
        this.alpha = alpha;
    }

    /**
     * 
     * @return the weight vector prior over theta
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Prior on topics. 1/{@link #setK(int) K} is  a common choice. 
     * @param eta the positive prior for topics
     */
    public void setEta(double eta)
    {
        if(eta <= 0 || Double.isInfinite(eta) || Double.isNaN(eta))
            throw new IllegalArgumentException("Eta must be a positive constant, not " + eta);
        this.eta = eta;
    }

    /**
     * 
     * @return the topic prior
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * A learning rate constant to control the influence of early iterations on 
     * the solution. Larger values reduce the influence of earlier iterations, 
     * smaller values increase the weight of earlier iterations. 
     * @param tau0 a learning rate parameter that must be greater than 0 (usually at least 1)
     */
    public void setTau0(double tau0)
    {
        if(tau0 <= 0 || Double.isInfinite(tau0) || Double.isNaN(tau0))
            throw new IllegalArgumentException("Eta must be a positive constant, not " + tau0);
        this.tau0 = tau0;
    }

    /**
     * Sets the number of training epochs when learning in a "batch" setting
     * @param epochs the number of iterations to go over the data set
     */
    public void setEpochs(int epochs)
    {
        this.epochs = epochs;
    }

    /**
     * Returns the number of training iterations over the data set that will be 
     * used
     * @return the number of training iterations over the data set that will be 
     * used
     */
    public int getEpochs()
    {
        return epochs;
    }
    
    /**
     * The "forgetfulness" factor in the learning rate. Larger values increase 
     * the rate at which old information is "forgotten" 
     * @param kappa the forgetfulness factor in [0.5, 1]
     */
    public void setKappa(double kappa)
    {
        if(kappa < 0.5 || kappa > 1.0 || Double.isNaN(kappa))
            throw new IllegalArgumentException("Kapp must be in [0.5, 1], not " + kappa);
        this.kappa = kappa;
    }

    /**
     * 
     * @return the forgetfulness factor
     */
    public double getKappa()
    {
        return kappa;
    }

    /**
     * Sets the number of data points used at a time to perform one update of 
     * the model parameters
     * @param miniBatchSize the batch size to use 
     */
    public void setMiniBatchSize(int miniBatchSize)
    {
        if(miniBatchSize < 1)
            throw new IllegalArgumentException("the batch size must be a positive constant, not " + miniBatchSize);
        this.miniBatchSize = miniBatchSize;
    }
    
    /**
     * Returns the topic vector for a given topic. The vector should not be 
     * altered, and is scaled so that the sum of all term weights sums to one. 
     * @param k the topic to get the vector for
     * @return the raw topic vector for the requested topic. 
     */
    public Vec getTopicVec(int k)
    {
        return new ScaledVector(1.0/lambda.get(k).sum(), lambda.get(k));
    }
    
    /**
     * From the 2013 paper, see expectations in figure 5 on page 1323, and 
     * equation (27) on page 1325
     * See also equation 6 in the 2010 paper. 2013 paper figure 5 seems to be a
     * typo
     * @param input the vector to take the input values from
     * @param sum the sum of the {@code input} vector
     * @param output the vector to store the transformed inputs in
     */
    private void expandPsiMinusPsiSum(Vec input, double sum, Vec output)
    {
        double psiSum = FastMath.digamma(sum);
        for(int i = 0; i < input.length(); i++)
            output.set(i, FastMath.digamma(input.get(i))-psiSum);
    }
    
    /**
     * Gets a sample from the exponential distribution. Implemented to be fast 
     * at the cost of accuracy
     * @param lambdaInv the inverse of the lambda value that parameterizes the 
     * exponential distribution
     * @param p the random value in [0, 1)
     * @return a sample from the exponential distribution
     */
    private static double sampleExpoDist(double lambdaInv, double p)
    {
        return -lambdaInv* FastMath.log(1-p);
    }
    
    /**
     * Performs an update of the LDA topic distributions based on the given 
     * mini-batch of documents.  
     * @param docs the list of document vectors to update from
     */
    public void update(List<Vec> docs)
    {
        update(docs, new FakeExecutor());
    }
    
    /**
     * Performs an update of the LDA topic distribution based on the given 
     * mini-batch of documents. 
     * @param docs the list of document vectors to update from
     * @param ex the source of threads for parallel execution
     */
    public void update(final List<Vec> docs, ExecutorService ex)
    {
        //need to init structure?
        if(lambda == null)
            initialize();
        /*
         * Make sure the beta values we will need are up to date
         */
        updateBetas(docs, ex);
        
        /*
         * Note, on each update we dont modify or access lambda untill the very,
         * end -  so we can interleave the "M" step into the final update 
         * accumulation to avoid temp space allocation and more easily exploit 
         * sparsity
         * 
         */
        
        
        //2: Set the step-size schedule ρt appropriately.
        final double rho_t = Math.pow(tau0+(t++), -kappa);
        
        
        //pre-shrink the lambda values so we can add out updates later
        for(int k = 0; k < K; k++)
        {
            lambda.get(k).mutableMultiply(1-rho_t);
            lambdaSums.set(k, lambdaSums.getD(k)*(1-rho_t));
        }
        
        /*
         * As described in the 2010 paper, this part is the "E" step if we view
         * it as an EM algorithm
         */
        
        //5: Initialize γ_dk =1, for k ∈ {1, . . . ,K}.
        
        /*
         * See note on page 3 from 2010 paper: "In practice, this algorithm 
         * converges to a better solution if we reinitialize γ and φ before
         * each E step"
         */
        final int P = SystemInfo.LogicalCores;
        final CountDownLatch latch = new CountDownLatch(P);
        //main iner loop, outer is per document and inner most is per topic convergence
        for(int id = 0; id < P; id++)
        {
            final int ID = id;
            ex.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    Random rand = RandomUtil.getRandom();
                    for(int d = ParallelUtils.getStartBlock(docs.size(), ID, P); d < ParallelUtils.getEndBlock(docs.size(), ID, P); d++)
                    {
                        final Vec doc = docs.get(d);
                        if(doc.nnz() == 0)
                            continue;
                        final Vec ELogTheta_d = logThetaLocal.get();
                        final Vec ExpELogTheta_d = expLogThetaLocal.get();
                        final Vec gamma_d = gammaLocal.get();

                        /*
                         * Make sure gamma and theta are set up and ready to start iterating 
                         */
                        prepareGammaTheta(gamma_d, ELogTheta_d, ExpELogTheta_d, rand);

                        int[] indexMap = new int[doc.nnz()];
                        double[] phiCols = new double[doc.nnz()];

                        //φ^k_dn ∝ exp{E[logθdk]+E[logβk,wdn ]}, k ∈ {1, . . . ,K}
                        computePhi(doc, indexMap, phiCols, K, gamma_d, ELogTheta_d, ExpELogTheta_d);

                        //accumulate updates, the "M" step
                        IntList toUpdate = new IntList(K);
                        ListUtils.addRange(toUpdate, 0, K, 1);
                        Collections.shuffle(toUpdate, rand);//helps reduce contention caused by shared iteration order
                        int updatePos = 0;
                        while(!toUpdate.isEmpty())
                        {
                            int k = toUpdate.getI(updatePos);
                            
                            if(lambdaLocks.get(k).tryLock())
                            {
                                final double coeff = ExpELogTheta_d.get(k)*rho_t*D/docs.size();
                                final Vec lambda_k = lambda.get(k);
                                final Vec ExpELogBeta_k = ExpELogBeta.get(k);
                                double lambdaSum_k = lambdaSums.getD(k);
                                
                                /*
                                 * iterate and incremebt ourselves so that we can also compute 
                                 * the new sums in 1 pass
                                 */
                                for(int i = 0; i < doc.nnz(); i++)
                                {
                                    int indx = indexMap[i];
                                    double toAdd = coeff*phiCols[i]*ExpELogBeta_k.get(indx);
                                    lambda_k.increment(indx, toAdd);
                                    lambdaSum_k += toAdd;
                                }

                                lambdaSums.set(k, lambdaSum_k);
                                lambdaLocks.get(k).unlock();
                                
                                toUpdate.remove(updatePos);
                            }
                            
                            if(!toUpdate.isEmpty())
                                updatePos = (updatePos+1) % toUpdate.size();
                        }
                    }
                    
                    latch.countDown();
                }
            });
            
        }
       
        try
        {
            latch.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(OnlineLDAsvi.class.getName()).log(Level.SEVERE, null, ex1);
        }
    }
    
    /**
     * Fits the LDA model against the given data set
     * @param dataSet the data set to learn a topic model for
     * @param topics the number of topics to learn 
     */
    public void model(DataSet dataSet, int topics)
    {
        model(dataSet, topics, new FakeExecutor());
    }
    
    /**
     * Fits the LDA model against the given data set
     * @param dataSet the data set to learn a topic model for
     * @param topics the number of topics to learn 
     * @param ex the source of threads for parallel execution
     */
    public void model(DataSet dataSet, int topics, ExecutorService ex)
    {
        if(ex == null)
            ex = new FakeExecutor();
        //Use notation same as original paper
        setK(topics);
        setD(dataSet.getSampleSize());
        setVocabSize(dataSet.getNumNumericalVars());
        
        final List<Vec> docs = dataSet.getDataVectors();
        
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(docs);
            for(int i = 0; i < D; i+=miniBatchSize)
            {
                int to = Math.min(i+miniBatchSize, D);
                update(docs.subList(i, to), ex);
            }
            
        }
    }

    /**
     * Computes the topic distribution for the given document.<br>
     * Note that the returned vector will be dense, but many of the values may 
     * be very nearly zero. 
     * 
     * @param doc the document to find the topics for
     * @return a vector of the topic distribution for the given document
     */
    public Vec getTopics(Vec doc)
    {
        Vec gamma = new DenseVector(K);

        Random rand = RandomUtil.getRandom();
        double lambdaInv = (W * K) / (D * 100.0);

        for (int j = 0; j < gamma.length(); j++)
            gamma.set(j, sampleExpoDist(lambdaInv, rand.nextDouble()) + eta);

        Vec eLogTheta_i = new DenseVector(K);
        Vec expLogTheta_i = new DenseVector(K);
        expandPsiMinusPsiSum(gamma, gamma.sum(), eLogTheta_i);
        for (int j = 0; j < eLogTheta_i.length(); j++)
            expLogTheta_i.set(j, FastMath.exp(eLogTheta_i.get(j)));
        
        computePhi(doc, new int[doc.nnz()], new double[doc.nnz()], K, gamma, eLogTheta_i, expLogTheta_i);
        gamma.mutableDivide(gamma.sum());
        return gamma;
    }
    
    /**
     * Updates the Beta vectors associated with the {@link #gammaLocal gamma} 
     * topic distributions so that they can be used to update against the given 
     * batch of documents. Once updated, the Betas are the only items needed to 
     * perform updates from the given batch, and the gamma values can be updated
     * as the updates are computed. 
     * 
     * @param docs the mini batch of documents to update from
     */
    private void updateBetas(final List<Vec> docs, ExecutorService ex)
    {
        final double[] digammaLambdaSum = new double[K];//TODO may want to move this out & reuse
        for(int k = 0; k < K; k++)
            digammaLambdaSum[k] = FastMath.digamma(W*eta+lambdaSums.getD(k));
        List<List<Vec>> docSplits = ListUtils.splitList(docs, SystemInfo.LogicalCores);
        final CountDownLatch latch = new CountDownLatch(docSplits.size());
        for(final List<Vec> docsSub :  docSplits)
        {
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    for(Vec doc : docsSub)//make sure out ELogBeta is up to date
                        for(IndexValue iv : doc)
                        {
                            int indx = iv.getIndex();
                            if(lastUsed[indx] != t)
                            {
                                for(int k = 0; k < K; k++)
                                {
                                    double lambda_kj = lambda.get(k).get(indx);

                                    double logBeta_kj = FastMath.digamma(eta+lambda_kj)-digammaLambdaSum[k];
                                    ELogBeta.get(k).set(indx, logBeta_kj);
                                    ExpELogBeta.get(k).set(indx, FastMath.exp(logBeta_kj));
                                }
                                lastUsed[indx] = t;
                            }
                        }
                    
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(OnlineLDAsvi.class.getName()).log(Level.SEVERE, null, ex1);
        }
    }

    /**
     * Prepares gamma and the associated theta expectations are initialized so 
     * that the iterative updates to them can begin. 
     * 
     * @param gamma_i will be completely overwritten 
     * @param eLogTheta_i will be completely overwritten 
     * @param expLogTheta_i will be completely overwritten 
     * @param rand the source of randomness
     */
    private void prepareGammaTheta(Vec gamma_i, Vec eLogTheta_i, Vec expLogTheta_i, Random rand)
    {
        final double lambdaInv = (W * K) / (D * 100.0);
        for (int j = 0; j < gamma_i.length(); j++)
            gamma_i.set(j, sampleExpoDist(lambdaInv, rand.nextDouble()) + eta);

        expandPsiMinusPsiSum(gamma_i, gamma_i.sum(), eLogTheta_i);
        for (int j = 0; j < eLogTheta_i.length(); j++)
            expLogTheta_i.set(j, FastMath.exp(eLogTheta_i.get(j)));
    }

    /**
     * Performs the main iteration to determine the topic distribution of the 
     * given document against the current model parameters. The non zero values 
     * of phi will be stored in {@code indexMap} and {@code phiCols}
     * 
     * @param doc the document to get the topic assignments for
     * @param indexMap the array of integers to store the non zero document 
     * indices in
     * @param phiCols the array to store the normalized non zero values of phi 
     * in, where each value corresponds to the associated index in 
     * {@code indexMap}
     * @param K the number of topics 
     * @param gamma_d the initial value of γ that will be altered to the topic assignments, but not normalized
     * @param ELogTheta_d the expectation from γ per topic
     * @param ExpELogTheta_d the exponentiated vector for {@code ELogTheta_d}
     */
    private void computePhi(final Vec doc, int[] indexMap, double[] phiCols, int K, final Vec gamma_d, final Vec ELogTheta_d, final Vec ExpELogTheta_d)
    {
        //φ^k_dn ∝ exp{E[logθdk]+E[logβk,wdn ]}, k ∈ {1, . . . ,K}
        /*
         * we have the exp versions of each, and exp(log(x)+log(y)) = x y
         * so we can just use the doc product between the vectors per 
         * document to get the normalization constan Z
         * 
         * When we update γ we multiply by the word, so non presnet words 
         * have no impact. So we don't need ALL of the columbs from φ, but
         * only the columns for which we have non zero words. 
         */

        /*
         * normalized for each topic column (len K) of the words in this doc.
         * We only need to concern oursleves with the non zeros
         * 
         * Beacse we need to update several iterations, we will work with 
         * the inernal stricture dirrectly instead of using expensitve 
         * get/set on a Sparse Vector
         */
        
        int pos = 0;
        final SparseVector updateVec = new SparseVector(indexMap, phiCols, W, doc.nnz());
        for(IndexValue iv : doc)
        {
            int wordIndex = iv.getIndex();
            double sum = 0;
            for(int i = 0; i < ExpELogTheta_d.length(); i++)
                sum += ExpELogTheta_d.get(i)*ExpELogBeta.get(i).get(wordIndex);

            indexMap[pos] = wordIndex;
            phiCols[pos] = iv.getValue()/(sum+1e-15);
            pos++;
        }
        //iterate till convergence or we hit arbitrary 100 limit (dont usually see more than 70)
        for(int iter = 0; iter < 100; iter++)
        {
            double meanAbsChange = 0;
            double gamma_d_sum = 0;
            //γtk = α+ w φ_twk n_tw
            for(int k = 0; k < K; k++)
            {
                final double origGamma_dk = gamma_d.get(k);
                double gamma_dtk = alpha;

                gamma_dtk += ExpELogTheta_d.get(k) * updateVec.dot(ExpELogBeta.get(k));
                gamma_d.set(k, gamma_dtk);
                meanAbsChange += Math.abs(gamma_dtk-origGamma_dk);
                gamma_d_sum += gamma_dtk;
            }
            
            //update Eq[log θtk] and our exponentated copy of it
            expandPsiMinusPsiSum(gamma_d, gamma_d_sum, ELogTheta_d);
            for(int i = 0; i < ELogTheta_d.length(); i++)
                ExpELogTheta_d.set(i, FastMath.exp(ELogTheta_d.get(i)));
            
            //update our column norm norms 
            int indx = 0;
            for(IndexValue iv : doc)
            {
                int wordIndex = iv.getIndex();
                double sum = 0;
                for(int i = 0; i < ExpELogTheta_d.length(); i++)
                    sum += ExpELogTheta_d.get(i)*ExpELogBeta.get(i).get(wordIndex);
                phiCols[indx] = iv.getValue() / (sum + 1e-15);
                indx++;
            }
            
            /*
             * //original papser uses a tighter bound, but our approximation
             * isn't that good - and this seems to work well enough
             * 0.01 even seems to work, but need to try that more before 
             * switching
             */
            if(meanAbsChange < 0.001*K)
                break;
        }
    }

    private void initialize()
    {
        if(K < 1)
            throw new FailedToFitException("Topic number for LDA has not yet been specified");
        else if(D < 1)
            throw new FailedToFitException("Expected number of documents has not yet been specified");
        else if(W < 1)
            throw new FailedToFitException("Topic vocuabulary size has not yet been specified");
        
        t = 0;
        //1: Initialize λ(0) randomly
        lambda = new ArrayList<Vec>(K);
        lambdaLocks = new ArrayList<Lock>(K);
        lambdaSums = new DoubleList(K);
        ELogBeta = new ArrayList<Vec>(K);
        ExpELogBeta = new ArrayList<Vec>(K);
        lastUsed = new int[W];
        Arrays.fill(lastUsed, -1);
        
        final double lambdaInv = (K*W)/(D*100.0);
        Random rand = RandomUtil.getRandom();
        for(int i = 0; i < K; i++)
        {
            Vec lambda_i = new DenseVector(W);
            lambda.add(new ScaledVector(lambda_i));
            lambdaLocks.add(new ReentrantLock());
            ELogBeta.add(new DenseVector(W));
            ExpELogBeta.add(new DenseVector(W));
            double rowSum = 0;
            for(int j = 0; j < W; j++)
            {
                double sample = sampleExpoDist(lambdaInv, rand.nextDouble())+eta;
                lambda_i.set(j, sample);
                rowSum += sample;
            }
            lambdaSums.add(rowSum);
        }
        //lambda has now been intialized, ELogBeta and ExpELogBeta will be intialized / updated lazily
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
