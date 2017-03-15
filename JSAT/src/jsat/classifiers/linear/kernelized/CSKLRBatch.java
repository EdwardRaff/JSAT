package jsat.classifiers.linear.kernelized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.svm.SupportVectorLearner;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * An implementation of Conservative Stochastic Kernel Logistic Regression. This
 * is an online algorithm that obtains sparse solutions by conservatively 
 * rejecting updates based on a binomial distribution of the error on each 
 * update. <br><br>
 * This algorithm works best on data sets with a very high number of samples 
 * where a high accuracy is obtainable using a kernel model. It is often the 
 * case that this model produces accurate results, but has a low confidence due 
 * to the conservative updating. This can be counteracted by having a very large
 * number of features, but that often increases the size of the model. 
 * <br>
 * This batch version can also be used to more efficiently learn dense KLR 
 * models using the stochastic method with the {@link CSKLR.UpdateMode#NC} mode if model 
 * sparsity is not important. 
 * <br><br>
 * It is important to read the documentation and test some different values for
 * the {@link #setEta(double) learning rate} and {@link #setGamma(double) gamma} 
 * variables. They behave different compared to many algorithms. 
 * <br><br>
 * See paper: <br>
 * Zhang, L., Jin, R., Chen, C., Bu, J.,&amp;He, X. (2012). <i>Efficient Online 
 * Learning for Large-Scale Sparse Kernel Logistic Regression</i>. Twenty-Sixth 
 * AAAI Conference on Artificial Intelligence (pp. 1219–1225). Retrieved from 
 * <a href="http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPDFInterstitial/5003/5544">here</a>
 * 
 * @author Edward Raff
 */
public class CSKLRBatch extends SupportVectorLearner implements Parameterized, Classifier
{
    private static final long serialVersionUID = -2305532659182911285L;
    
    private double eta;
    private double curNorm;
    private double R = 10;
    private int T = 0;
    private CSKLR.UpdateMode mode;
    protected double gamma = 2;
    private int epochs = 10;

    /**
     * Creates a new SCKLR Batch learning object
     * @param eta the learning rate to use
     * @param kernel the kernel to use
     * @param R the maximal norm of the surface 
     * @param mode the mode to use
     * @param cacheMode the kernel caching mode to use
     */
    public CSKLRBatch(double eta, KernelTrick kernel, double R, CSKLR.UpdateMode mode, CacheMode cacheMode)
    {
        super(kernel, cacheMode);
        setEta(eta);
        setR(R);
        setMode(mode);
    }

    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    protected CSKLRBatch(CSKLRBatch toClone)
    {
        super(toClone);
        
        this.curNorm = toClone.curNorm;
        this.epochs = toClone.epochs;
        this.eta = toClone.eta;
        this.R = toClone.R;
        this.T = toClone.T;
        this.mode = toClone.mode;
        this.gamma = toClone.gamma;
        
    }

    @Override
    public CSKLRBatch clone()
    {
        return new CSKLRBatch(this);
    }

    /**
     * Sets the number of training epochs (passes) through the data set
     * @param epochs the number of passes through the data set
     */
    public void setEpochs(int epochs)
    {
        this.epochs = epochs;
    }

    /**
     * Returns the number of passes through the data set
     * @return the number of passes through the data set
     */
    public int getEpochs()
    {
        return epochs;
    }

    /**
     * Sets the learning rate to use for the algorithm. Unlike many other 
     * stochastic algorithms, the learning rate for CSKLR should be large, often
     * in the range of (0.5, 1) - and can even be larger than 1 at times. If the
     * learning rate is too low, it may be difficult to get strong confidence 
     * results from the algorithm. 
     * 
     * @param eta the positive learning rate to use
     */
    public void setEta(double eta)
    {
        if(eta < 0 || Double.isNaN(eta) || Double.isInfinite(eta))
            throw new IllegalArgumentException("The learning rate should be in (0, Inf), not " + eta);
        this.eta = eta;
    }

    /**
     * Returns the learning rate to use
     * @return the learning rate to use
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets the maximal margin norm value for the algorithm. When the norm is 
     * exceeded, the coefficients will be rescaled to fit in the norm. If the 
     * maximal norm is too small (less than 5), it may be difficult to get 
     * strong confidence results from the algorithm. <br>
     * A good range of values suggested by the original paper is 10<sup>x</sup>
     * &forall; x &isin; {0, 1, 2, 3, 4, 5}
     * @param R 
     */
    public void setR(double R)
    {
        if(R < 0 || Double.isNaN(R) || Double.isInfinite(R))
            throw new IllegalArgumentException("The max norm should be in (0, Inf), not " + R);
        this.R = R;
    }

    /**
     * Returns the maximal norm of the algorithm
     * @return the maximal norm of the algorithm
     */
    public double getR()
    {
        return R;
    }

    /**
     * Sets what update mode should be used. The update mode controls the 
     * sparsity of the mode, and the behavior of {@link #setGamma(double) }
     * @param mode the update mode to use
     */
    public void setMode(CSKLR.UpdateMode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns the update mode in use
     * @return  the update mode in use
     */
    public CSKLR.UpdateMode getMode()
    {
        return mode;
    }

    /**
     * Sets the gamma value to use. This value, depending on which 
     * {@link CSKLR.UpdateMode} is used, controls the sparsity of the model.
     * @param gamma the gamma parameter, which is at least always positive
     */
    public void setGamma(double gamma)
    {
        if(gamma < 0 || Double.isNaN(gamma) || Double.isInfinite(gamma))
            throw new IllegalArgumentException("Gamma must be in (0, Infity), not " + gamma);
        this.gamma = gamma;
    }

    /**
     * Returns the gamma sparsity parameter value
     * @return the gamma sparsity parameter value
     */
    public double getGamma()
    {
        return gamma;
    }
    
    /**
     * Guesses the distribution to use for the R parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the R parameter
     * @see #setR(double) 
     */
    public static Distribution guessR(DataSet d)
    {
        return new LogUniform(1, 1e5);
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

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        
        double p_0 = CSKLR.getScore(-1, getPreScore(data.getNumericalValues()));

        cr.setProb(0, p_0);
        cr.setProb(1, 1-p_0);
        
        return cr;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("CSKLR supports only binary classification");
        //First we need to set up the vectors array

        final int N = dataSet.getSampleSize();
        vecs = new ArrayList<Vec>(N);
        alphas = new double[N];
        for(int i = 0; i < N; i++)
            vecs.add(dataSet.getDataPoint(i).getNumericalValues());
        
        curNorm = 0;
        T = 0;
        Random rand = RandomUtil.getRandom();
        
        IntList sampleOrder = new IntList(N);
        ListUtils.addRange(sampleOrder, 0, N, 1);
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(sampleOrder);
            
            for(int i : sampleOrder)
            {
                final double weight = dataSet.getDataPoint(i).getWeight();
                final double y_t = dataSet.getDataPointCategory(i)*2-1;
                final Vec x_t = vecs.get(i);
                final double pre = getPreScore(x_t);
                final double score = CSKLR.getScore(y_t, pre);

                switch(mode)
                {
                    case NC:
                        break;
                    default:
                        double pt = mode.pt(y_t, score, pre, eta, gamma);
                        if(rand.nextDouble() > pt)
                            continue;
                     break;   
                }


                double alpha_i = -eta*y_t*mode.grad(y_t, score, pre, gamma)*weight;
        
                alphas[i] += alpha_i;

                curNorm += Math.abs(alpha_i)*kEval(i, i);

                //projection step
                if(curNorm > R)
                {
                    double coef = R/curNorm;
                    for(int j = 0; j < alphas.length; j++)
                        alphas[j] *= coef;
                    curNorm = coef;
                }
            }
            
        }
        
        int supportVectorCount = 0;
        for(int i = 0; i < N; i++)
            if(alphas[i] > 0 || alphas[i] < 0)//Its a support vector
            {
                ListUtils.swap(vecs, supportVectorCount, i);
                alphas[supportVectorCount++] = alphas[i];
            }
        vecs = new ArrayList<Vec>(vecs.subList(0, supportVectorCount));
        alphas = Arrays.copyOfRange(alphas, 0, supportVectorCount);
        
        setCacheMode(null);
        setAlphas(alphas);
    }
    
    private double getPreScore(Vec x)
    {
        return kEvalSum(x);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
}
