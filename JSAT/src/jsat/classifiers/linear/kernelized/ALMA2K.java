package jsat.classifiers.linear.kernelized;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.classifiers.linear.ALMA2;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.IntList;

/**
 * Provides a kernelized version of the {@link ALMA2} algorithm. It is important
 * to note that the number of "support vectors" ALMA may learn is unbounded.
 * <br>
 * The averaged output of all previous hyperplanes is supported at almost no 
 * overhead, and can be turned on by setting {@link #setAveraged(boolean) }. 
 * This information is always collected, and the output can be changed once 
 * already learned. 
 * <br><br>
 * See: Gentile, C. (2002). <i>A New Approximate Maximal Margin Classification 
 * Algorithm</i>. The Journal of Machine Learning Research, 2, 213–242. 
 * Retrieved from <a href="http://dl.acm.org/citation.cfm?id=944811">here</a>
 * 
 * @author Edward Raff
 */
public class ALMA2K extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized
{
    private static final long serialVersionUID = 7247320234799227009L;
    
    private static final double p = 2;
    private double alpha;
    private double B;
    private double C = Math.sqrt(2);
    private int k;
    private int curRounds;
    @ParameterHolder
    private KernelTrick K;
    private List<Vec> supports;
    private List<Double> signedEtas;
    private List<Double> associatedScores;
    private List<Double> normalizers;
    private List<Integer> rounds;
    
    private boolean averaged = false;

    /**
     * Creates a new kernelized ALMA2 object
     * @param kernel the kernel function to use
     * @param alpha the alpha parameter of ALMA
     */
    public ALMA2K(KernelTrick kernel,  double alpha)
    {
        setKernelTrick(kernel);
        setAlpha(alpha);
    }

    /**
     * Copy constructor
     * @param other the ALMA2K object to copy
     */
    protected ALMA2K(ALMA2K other)
    {
        this.alpha = other.alpha;
        this.B = other.B;
        this.C = other.C;
        this.k = other.k;
        this.K = other.K.clone();
        this.averaged = other.averaged;
        
        if(other.supports != null)
        {
            this.supports = new ArrayList<Vec>(other.supports.size());
            for(Vec v : other.supports)
                this.supports.add(v.clone());
            this.signedEtas = new DoubleList(other.signedEtas);
            this.associatedScores = new DoubleList(other.associatedScores);
            this.normalizers = new DoubleList(other.normalizers);
            this.rounds = new IntList(other.rounds);
        }
    }
    
    @Override
    public ALMA2K clone()
    {
        return new ALMA2K(this);
    }

    /**
     * ALMA2K supports taking the averaged output of all previous hypothesis 
     * weighted by the number of successful uses of the hypothesis during 
     * training. This effectively reduces the variance of the classifier. It has
     * no impact on the training / update phase, only the classification results 
     * are impacted. 
     * <br><br>
     * Unlike most algorithms, this can be changed at any time without issue - 
     * even after the algorithm has been trained the type of output (averaged or
     * last) can be switched on the fly. 
     * @param averaged {@code true} to use the averaged out, {@code false} to 
     * only use the last hypothesis 
     */
    public void setAveraged(boolean averaged)
    {
        this.averaged = averaged;
    }

    /**
     * Returns whether or not the averaged or last hypothesis is used
     * @return whether or not the averaged or last hypothesis is used
     */
    public boolean isAveraged()
    {
        return averaged;
    }

    /**
     * Sets the kernel to use 
     * @param K the kernel to use 
     */
    public void setKernelTrick(KernelTrick K)
    {
        this.K = K;
    }

    /**
     * Returns the kernel in use
     * @return the kernel in use
     */
    public KernelTrick getKernelTrick()
    {
        return K;
    }
    
    /**
     * Alpha controls the approximation of the large margin formed by ALMA, 
     * with larger values causing more updates. A value of 1.0 will update only
     * on mistakes, while smaller values update if the error was not far enough
     * away from the margin. 
     * <br><br>
     * NOTE: Whenever alpha is set, the value of {@link #setB(double) B} will 
     * also be set to an appropriate value. This is not the only possible value 
     * that will lead to convergence, and can be set manually after alpha is set
     * to another value. 
     * 
     * @param alpha the approximation scale in (0.0, 1.0]
     */
    public void setAlpha(double alpha)
    {
        if(alpha <= 0 || alpha > 1 || Double.isNaN(alpha))
            throw new ArithmeticException("alpha must be in (0, 1], not " + alpha);
        this.alpha = alpha;
        setB(1.0/alpha);
    }

    /**
     * Returns the approximation coefficient used  
     * @return the approximation coefficient used 
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Sets the B variable of the ALMA algorithm, this is set automatically by 
     * {@link #setAlpha(double) }. 
     * @param B the value for B
     */
    public void setB(double B)
    {
        this.B = B;
    }

    /**
     * Returns the B value of the ALMA algorithm
     * @return the B value of the ALMA algorithm
     */
    public double getB()
    {
        return B;
    }

    /**
     * Sets the C value of the ALMA algorithm. The default value is the one 
     * suggested in the paper. 
     * @param C the C value of ALMA
     */
    public void setC(double C)
    {
        if(C <= 0 || Double.isInfinite(C) || Double.isNaN(C))
            throw new ArithmeticException("C must be a posative cosntant");
        this.C = C;
    }

    public double getC()
    {
        return C;
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes <= 0)
            throw new FailedToFitException("ALMA2 requires numeric features");
        if(predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("ALMA2 works only for binary classification");
        
        supports = new ArrayList<Vec>();
        signedEtas = new DoubleList();
        associatedScores = new DoubleList();
        normalizers = new DoubleList();
        rounds = new IntList();
        k = 1;
        curRounds = 0;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final double y_t = targetClass*2-1;
        
        double gamma = B * Math.sqrt(p-1) / k;
        double wx = score(x_t, false);
        if(y_t*wx <= (1-alpha)*gamma)//update
        {
            double eta = C/Math.sqrt(p-1)/Math.sqrt(k++);
            
            double norm = Math.sqrt(K.eval(x_t, x_t));
            
            associatedScores.add(score(new ScaledVector(1/norm, x_t), false));
            supports.add(x_t);
            normalizers.add(norm);
            signedEtas.add(eta*y_t);
            rounds.add(curRounds);
            curRounds = 0;
        }
        else
            curRounds++;
    }
    
    /**
     * Computes the output of the summations of the input vector with the 
     * current weight vector as a recursive linear combination of all previous 
     * support vectors and their associated score values. <br>
     * See Remark 5 in the original paper. 
     * @param x the input vector to compute the score value
     * @return the score for the input indicating which side of the hyperplane
     * it is on
     */
    private double score(Vec x, boolean averaged)
    {
        /*
         * Score for the current dot procut with the weight vector, denom for
         * the current normalizing constant. 
         */
        double score = 0;
        double denom = 0;
        double finalScore = 0;

        for(int i = 0; i < supports.size(); i++)
        {
            double eta_s = signedEtas.get(i);
            double tmp = eta_s*K.eval(supports.get(i), x)/normalizers.get(i);
            double denom_tmp = 2*eta_s*associatedScores.get(i)+eta_s*eta_s;
            denom += denom/Math.max(1, denom)+ denom_tmp;
            score += tmp/Math.max(1, denom);
            if(averaged)
                finalScore += score*rounds.get(i);
        }
        if(averaged)
            return finalScore;
        else
            return score;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        double wx = getScore(data);
        CategoricalResults cr =new CategoricalResults(2);
        if(wx < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return score(dp.getNumericalValues(), averaged);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
    /**
     * Guesses the distribution to use for the &alpha; parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the &alpha; parameter
     * @see #setAlpha(double)
     */
    public static Distribution guessAlpha(DataSet d)
    {
        return new Uniform(1e-3, 1.0);
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
