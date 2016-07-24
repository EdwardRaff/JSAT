package jsat.datatransform;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.*;
import static java.lang.Math.*;
import jsat.exceptions.FailedToFitException;

/**
 * Provides an implementation of the FastICA algorithm for Independent Component
 * Analysis (ICA). ICA is similar to PCA and Whitening, but assumes that the 
 * data is generated from a mixture of some <i>C</i> base components where 
 * mixing occurs instantaneously (i.e. produced from some matrix transform of 
 * the true components). ICA attempts to find the <i>C</i> components from the 
 * raw observations. <br>
 * <br>
 * See:
 * <ul>
 * <li>Hyvärinen, A. (1999). <i>Fast and robust fixed-point algorithms for 
 * independent component analysis</i>. IEEE Transactions on Neural Networks 
 * / a Publication of the IEEE Neural Networks Council, 10(3), 626–34. 
 * doi:10.1109/72.761722
 * </li>
 * <li>
 * Hyvärinen, a.,&amp;Oja, E. (2000). <i>Independent component analysis: 
 * algorithms and applications</i>. Neural Networks, 13(4-5), 411–430. 
 * doi:10.1016/S0893-6080(00)00026-5
 * </li>
 * </ul>
 * @author Edward Raff
 */
public class FastICA implements InvertibleTransform
{
    //TODO add default search for C
    private static final long serialVersionUID = -8644025740457515563L;

    /**
     * the number of base components to assume and try to discover
     */
    private int C;
    /**
     * the Negative Entropy function to use
     */
    private NegEntropyFunc G;
    /**
     * {@code true} to assume the data has already been 
     */
    private boolean preWhitened;
    private ZeroMeanTransform zeroMean;

    /**
     * Un-mixes the observed data into the raw components we learned 
     */
    private Matrix unmixing;
    /**
     * The estimated mixing matrix to go from raw components to the observed data
     */
    private Matrix mixing;

    /**
     * The FastICA algorithm requires a function f(x) to be used iteratively in 
     * the algorithm, but only makes use of the first and second derivatives of
     * the algorithm. 
     */
    public static interface NegEntropyFunc
    {
        /**
         * 
         * @param x the input to the function
         * @return the first derivative of this function
         */
        public double deriv1(double x);
        
        /**
         * 
         * @param x the input to the function
         * @param d1 the first derivative of this function (from
         * {@link #deriv1(double) })
         * @return the second derivative of this function
         */
        public double deriv2(double x, double d1);
    }
    
    /**
     * A set of default negative entropy functions as specified in the original 
     * FastICA paper
     */
    public enum DefaultNegEntropyFunc implements NegEntropyFunc
    {
        /**
         * This is function <i>G<sub>1</sub></i> in the paper. This Negative 
         * Entropy function is described as a "good general-purpose contrast 
         * function" in the original paper, and the default method used. 
         */
        LOG_COSH 
        {

            @Override
            public double deriv1(double x)
            {
                return tanh(x);
            }

            @Override
            public double deriv2(double x, double d1)
            {
                return 1-d1*d1;
            }
        },
        /**
         * This is function <i>G<sub>2</sub></i> in the paper, and according to 
         * the paper may be better than {@link #LOG_COSH}   "when the 
         * independent components are highly super-Gaussian, or when 
         * robustness is very important"
         */
        EXP 
        {
            @Override
            public double deriv1(double x)
            {
                return x*exp(-x*x/2);
            }

            @Override
            public double deriv2(double x, double d1)
            {
                //calling exp is more expensive than just dividing to get back e(-x^2/2)
                if(x == 0)
                    return 1;
                return (1-x*x)*(d1/x);
            }
        },
        /**
         * This is the kurtosis-based approximation function <i>G<sub>3</sub>(x)
         * = 1/4*x<sup>4</sup></i>. According to the original paper its use is 
         * "is justified on statistical grounds only for estimating sub-Gaussian
         * independent components when there are no outliers."
         */
        KURTOSIS
        {
            @Override
            public double deriv1(double x)
            {
                return x*x*x;//x^3
            }

            @Override
            public double deriv2(double x, double d1)
            {
                return x*x*3;//3 x^2
            }
        };

        @Override
        abstract public double deriv1(double x);

        @Override
        abstract public double deriv2(double x, double d1);
    };
    
    /**
     * Creates a new FastICA transform that will attempt to fit 10 components.
     * This is likely not optimal for any particular dataset
     */
    public FastICA()
    {
        this(10);
    }
    
    /**
     * Creates a new FastICA transform
     * 
     * @param C the number of base components to assume and try to discover
     */
    public FastICA(int C)
    {
        this(C, DefaultNegEntropyFunc.LOG_COSH, false);
    }
    
    /**
     * Creates a new FastICA transform
     * @param data the data set to transform
     * @param C the number of base components to assume and try to discover
     */
    public FastICA(DataSet data, int C)
    {
        this(data, C, DefaultNegEntropyFunc.LOG_COSH, false);
    }

    /**
     * Creates a new FastICA transform
     *
     * @param data the data set to transform
     * @param C the number of base components to assume and try to discover
     * @param G the Negative Entropy function to use
     * @param preWhitened {@code true} to assume the data has already been 
     * whitened before being given to the transform, {@code false} and the 
     * FastICA implementation will perform its own whitening. 
     */
    public FastICA(int C, NegEntropyFunc G, boolean preWhitened)
    {
        setC(C);
        setNegEntropyFunction(G);
        setPreWhitened(preWhitened);
    }
    
    /**
     * Creates a new FastICA transform
     * @param data the data set to transform
     * @param C the number of base components to assume and try to discover
     * @param G the Negative Entropy function to use
     * @param preWhitened {@code true} to assume the data has already been 
     * whitened before being given to the transform, {@code false} and the 
     * FastICA implementation will perform its own whitening. 
     */
    public FastICA(DataSet data, int C, NegEntropyFunc G, boolean preWhitened)
    {
        this(C, G, preWhitened);
        fit(data);
    }

    @Override
    public void fit(DataSet data)
    {
        int N = data.getSampleSize();
        
        Vec tmp = new DenseVector(N);
             
        List<Vec> ws = new ArrayList<Vec>(C);
        
        Matrix X;
        WhitenedPCA whiten = null;
        
        if(!preWhitened)
        {
            //well allocate a dense matrixa and grab row view for extra efficency 
            zeroMean = new ZeroMeanTransform(data);
            data = data.shallowClone();
            data.applyTransform(zeroMean);
            
            whiten = new WhitenedPCA(data);
            
            data.applyTransform(whiten);
            X = data.getDataMatrixView();
        }
        else
            X = data.getDataMatrixView();
        
        int subD = X.cols();//projected space may be smaller if low rank
        Vec w_tmp = new DenseVector(subD);//used to check for convergence
        
        int maxIter = 500;//TODO make this configurable
                
        for(int  p  = 0; p < C; p++)
        {
            Vec w_p = Vec.random(subD);
            w_p.normalize();
            
            int iter = 0;
            
            do
            {
                //w_tmp is our old value use for convergence checking
                w_p.copyTo(w_tmp);
                
                
                tmp.zeroOut();
                X.multiply(w_p, 1.0, tmp);

                double gwx_avg = 0;
                for(int i = 0; i < tmp.length(); i++)
                {
                    final double x = tmp.get(i);
                    final double g = G.deriv1(x);
                    final double gp = G.deriv2(x, g);
                    if(Double.isNaN(g) || Double.isInfinite(g) || 
                            Double.isNaN(gp) || Double.isNaN(gp))
                        throw new FailedToFitException("Encountered NaN or Inf in calculation");
                    tmp.set(i, g);
                    gwx_avg += gp;
                }

                gwx_avg /= N;
                
                //w+ =E{xg(wTx)}−E{g'(wT x)}w
                w_p.mutableMultiply(-gwx_avg);
                X.transposeMultiply(1.0/N, tmp, w_p);
                
                //reorthoganalization by w_p = w_p - sum_{i=0}^{p-1} w_p^T w_j w_j 
                double[] coefs = new double[ws.size()];
                for(int i= 0; i < coefs.length; i++)
                    coefs[i] = w_p.dot(ws.get(i));
                for(int i= 0; i < coefs.length; i++)
                    w_p.mutableAdd(-coefs[i], ws.get(i));
                
                //re normalize
                w_p.normalize();
                
                
                /*
                 * Convergence check at end of loop: "Note that convergencemeans
                 * that the old and new values of w point in the same direction,
                 * i.e. their dot-product is (almost) equal to 1. It is not 
                 * necessary that the vector converges to a single point, since 
                 * w and −w define the same direction"
                 */
            }
            while(abs(1-abs(w_p.dot(w_tmp))) > 1e-6 && iter++ < maxIter);
            
            ws.add(w_p);
            
        }
        
        
        if(!preWhitened)
        {
            Matrix W = new MatrixOfVecs(ws);
            
            unmixing = W.multiply(whiten.transform).transpose();
        }
        else
            unmixing = new DenseMatrix(new MatrixOfVecs(ws)).transpose();
        
        mixing = new SingularValueDecomposition(unmixing.clone()).getPseudoInverse();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public FastICA(FastICA toCopy)
    {
        this.C = toCopy.C;
        this.G = toCopy.G;
        this.preWhitened = toCopy.preWhitened;
        if (toCopy.zeroMean != null)
            this.zeroMean = toCopy.zeroMean.clone();
        if (toCopy.unmixing != null)
            this.unmixing = toCopy.unmixing.clone();
        if (toCopy.mixing != null)
            this.mixing = toCopy.mixing.clone();
    }
    
    /**
     * Sets the number of base components to learn
     * @param C the number of base components to assume and try to discover
     */
    public void setC(int C)
    {
        if(C < 1)
            throw new IllegalArgumentException("Number of components must be positive, not " + C);
        this.C = C;
    }

    /**
     * 
     * @return the number of base components to assume and try to discover
     */
    public int getC()
    {
        return C;
    }
    
    /**
     * Sets the Negative Entropy function used to infer the base components. 
     * 
     * @param G the Negative Entropy function to use
     */
    public void setNegEntropyFunction(NegEntropyFunc G)
    {
        if(G == null)
            throw new NullPointerException("Negative Entropy function must be non-null");
        this.G = G;
    }
    
    /**
     * 
     * @return the Negative Entropy function to use
     */
    public NegEntropyFunc getNegEntropyFunction()
    {
        return G;
    }

    /**
     * Controls where or not the implementation assumes the input data is
     * already whitened. Whitening is a requirement for the algorithm to work as
     * intended.
     *
     * @param preWhitened {@code true} to assume the data has already been
     * whitened, {@code false} for this object to do its own whitening
     */
    public void setPreWhitened(boolean preWhitened)
    {
        this.preWhitened = preWhitened;
    }

    /**
     *
     * @return {@code true} if this object will assume the data has already been
     * whitened, {@code false} for this object to do its own whitening
     */
    public boolean isPreWhitened()
    {
        return preWhitened;
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec x;
        if (zeroMean != null)
            x = zeroMean.transform(dp).getNumericalValues();
        else
            x = dp.getNumericalValues();

        Vec newX = x.multiply(unmixing);

        //we know that zeroMean wont impact cat values or weight
        return new DataPoint(newX, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }
    
    @Override
    public DataPoint inverse(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        x = x.multiply(mixing);
        
        DataPoint toRet = new DataPoint(x, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
        if(zeroMean != null)
            zeroMean.mutableInverse(toRet);
        
        return toRet;
    }

    @Override
    public FastICA clone()
    {
        return new FastICA(this);
    }
}
