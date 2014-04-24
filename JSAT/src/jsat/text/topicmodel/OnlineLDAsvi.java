package jsat.text.topicmodel;

import java.util.*;
import jsat.DataSet;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.math.FastMath;
import jsat.utils.DoubleList;
import jsat.utils.random.XOR96;
import jsat.utils.random.XORWOW;

/**
 * This class provides an implementation of <i>Latent Dirichlet Allocation</i>
 * for learning a topic model from a set of documents. This implementation is 
 * based on Stochastic Variational Inference and is meant for large collections 
 * (more than 100,000 data points) and can learn in an online fashion. <br>
 * <br>
 * See:<br>
 * <ul>
 * <li>Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). <i>Latent Dirichlet 
 * Allocation</i>. Journal of Machine Learning Research, 3(4-5), 993–1022.
 * doi:10.1162/jmlr.2003.3.4-5.993</li>
 * <li>Hoffman, M., Blei, D., & Bach, F. (2010). <i>Online Learning for Latent 
 * Dirichlet Allocation</i>. In Advances in Neural Information Processing 
 * Systems (pp. 856–864). Retrieved from 
 * <a href="http://videolectures.net/site/normal_dl/tag=83534/nips2010_1291.pdf">
 * here</a></li>
 * <li>Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). 
 * <i>Stochastic Variational Inference</i>. The Journal of Machine Learning 
 * Research, 14(1), 1303–1347.</li>
 * <li>Hoffman, M. D. (2013). <i>Lazy updates for online LDA</i>. Retrieved from
 * <a href="https://groups.yahoo.com/neo/groups/vowpal_wabbit/conversations/topics/250">here</a></li>
 * </ul>
 * @author Edward Raff
 */
public class OnlineLDAsvi
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
    public List<Vec> lambda;
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
     * Sets the prior for the on weight vector theta
     * @param alpha 
     */
    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
    }

    /**
     * Prior on topics
     * @param eta 
     */
    public void setEta(double eta)
    {
        this.eta = eta;
    }

    public void setTau0(double tau0)
    {
        this.tau0 = tau0;
    }

    public void setEpochs(int epochs)
    {
        this.epochs = epochs;
    }
    
    /**
     * Forgetfullness 
     * @param kappa 
     */
    public void setKappa(double kappa)
    {
        this.kappa = kappa;
    }

    public void setMiniBatchSize(int miniBatchSize)
    {
        this.miniBatchSize = miniBatchSize;
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
        //need to init structure?
        if(lambda == null)
            initialize();
        /*
         * Make sure the beta values we will need are up to date
         */
        updateBetas(docs);
        
        /*
         * Note, on each update we dont modify or access lambda untill the very,
         * end -  so we can interleave the "M" step into the final update 
         * accumulation to avoid temp space allocation and more easily exploit 
         * sparsity
         * 
         */
        
        
        //2: Set the step-size schedule ρt appropriately.
        double rho_t = Math.pow(tau0+(t++), -kappa);
        
        
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
        Random rand = new XORWOW();
        /*
         * See note on page 3 from 2010 paper: "In practice, this algorithm 
         * converges to a better solution if we reinitialize γ and φ before
         * each E step"
         */
        
        //main iner loop, outer is per document and inner most is per topic convergence
        for(int d = 0; d < docs.size(); d++)
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
            for(int k = 0; k < K; k++)
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
            }   
        }
    }
    
    /**
     * First the LDA model against the given data set
     * @param dataSet the data set to learn a topic model for
     * @param topics the number of topics to learn 
     */
    public void model(DataSet dataSet, int topics)
    {
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
                update(docs.subList(i, to));
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

        Random rand = new XOR96();
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
    private void updateBetas(List<Vec> docs)
    {
        final double[] digammaLambdaSum = new double[K];//TODO may want to move this out & reuse
        for(int k = 0; k < K; k++)
            digammaLambdaSum[k] = FastMath.digamma(W*eta+lambdaSums.getD(k));
        for(Vec doc : docs)//make sure out ELogBeta is up to date
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
        lambdaSums = new DoubleList(K);
        ELogBeta = new ArrayList<Vec>(K);
        ExpELogBeta = new ArrayList<Vec>(K);
        lastUsed = new int[W];
        Arrays.fill(lastUsed, -1);
        
        final double lambdaInv = (K*W)/(D*100.0);
        Random rand = new XORWOW();
        for(int i = 0; i < K; i++)
        {
            Vec lambda_i = new DenseVector(W);
            lambda.add(new ScaledVector(lambda_i));
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
}
