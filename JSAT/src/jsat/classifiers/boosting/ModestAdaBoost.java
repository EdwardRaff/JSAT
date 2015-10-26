package jsat.classifiers.boosting;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;

/**
 * Modest Ada Boost is a generalization of Discrete Ada Boost that attempts to 
 * reduce the generalization error and avoid over-fitting. Empirically, 
 * ModestBoost usually maintains a higher training-set error, and may take more
 * iterations to obtain the same test set error as other algorithms, but doesn't
 * not increase as much after it reaches the minimum error - which should make 
 * it easier to obtain the higher accuracy.
 * <br>
 * See: <br>
 * Vezhnevets, A.,&amp;Vezhnevets, V. (2005). <i>“Modest AdaBoost” – Teaching 
 * AdaBoost to Generalize Better</i>. GraphiCon. Novosibirsk Akademgorodok, 
 * Russia. Retrieved from 
 * <a href="http://www.inf.ethz.ch/personal/vezhneva/Pubs/ModestAdaBoost.pdf">
 * here</a>
 * 
 * @author Edward Raff
 */
public class ModestAdaBoost  implements Classifier, Parameterized, BinaryScoreClassifier
{

	private static final long serialVersionUID = 8223388561185098909L;
	private Classifier weakLearner;
    private int maxIterations;
    /**
     * The list of weak hypothesis
     */
    protected List<Classifier> hypoths;
    /**
     * The weights for each weak learner
     */
    protected List<Double> hypWeights;
    protected CategoricalData predicting;

    /**
     * Creates a new ModestBoost learner
     * @param weakLearner the weak learner to use
     * @param maxIterations the maximum number of boosting iterations
     */
    public ModestAdaBoost(final Classifier weakLearner, final int maxIterations)
    {
        setWeakLearner(weakLearner);
        setMaxIterations(maxIterations);
    }
    
    /**
     * Copy constructor
     * @param toClone the object to clone
     */
    protected ModestAdaBoost(final ModestAdaBoost toClone)
    {
        this(toClone.weakLearner.clone(), toClone.maxIterations);
        if(toClone.hypWeights != null)
        {
            this.hypWeights = new DoubleList(toClone.hypWeights);
            this.hypoths = new ArrayList<Classifier>(toClone.maxIterations);
            for(final Classifier weak : toClone.hypoths) {
              this.hypoths.add(weak.clone());
            }
            this.predicting = toClone.predicting.clone();
        }
    }
    
    /**
     * Returns the maximum number of iterations used
     * @return the maximum number of iterations used
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets the maximal number of boosting iterations that may be performed 
     * @param maxIterations the maximum number of iterations
     */
    public void setMaxIterations(final int maxIterations)
    {
        if(maxIterations < 1) {
          throw new IllegalArgumentException("Iterations must be positive, not " + maxIterations);
        }
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the weak learner currently being used by this method. 
     * @return the weak learner currently being used by this method. 
     */
    public Classifier getWeakLearner()
    {
        return weakLearner;
    }

    /**
     * Sets the weak learner used during training. 
     * @param weakLearner the weak learner to use
     */
    public void setWeakLearner(final Classifier weakLearner)
    {
        if(!weakLearner.supportsWeightedData()) {
          throw new IllegalArgumentException("WeakLearner must support weighted data to be boosted");
        }
        this.weakLearner = weakLearner;
    }
    
    @Override
    public double getScore(final DataPoint dp)
    {
        double score = 0;
        for(int i = 0; i < hypoths.size(); i++) {
          score += (hypoths.get(i).classify(dp).getProb(1)*2-1)*hypWeights.get(i);
        }
        return score;
    }

    @Override
    public CategoricalResults classify(final DataPoint data)
    {
        if(predicting == null) {
          throw new RuntimeException("Classifier has not been trained yet");
        }
        
        final CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        final double score =  getScore(data);
        if(score < 0) {
          cr.setProb(0, 1.0);
        } else {
          cr.setProb(1, 1.0);
        }
        return cr;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        predicting = dataSet.getPredicting();
        hypWeights = new DoubleList(maxIterations);
        hypoths = new ArrayList<Classifier>(maxIterations);
        final int N = dataSet.getSampleSize();
        
        final double[] D_inv = new double[N];
        final double[] D = new double[N];
        
        final List<DataPointPair<Integer>> dataPoints = dataSet.getTwiceShallowClone().getAsDPPList();
        Arrays.fill(D, 1.0/N);
        for(final DataPointPair<Integer> dpp : dataPoints) {
          dpp.getDataPoint().setWeight(D[0]);//Scaled, they are all 1 
        }
        double weightSum = 1;
        
        final double[] H_cur = new double[N];
        
        for(int t = 0; t < maxIterations; t++)
        {
            final Classifier weak = weakLearner.clone();
            if(threadPool == null || threadPool instanceof FakeExecutor) {
              weak.trainC(new ClassificationDataSet(dataPoints, predicting));
            } else {
              weak.trainC(new ClassificationDataSet(dataPoints, predicting), threadPool);
            }
            
            double invSum = 0;
            for(int i = 0; i < N; i++) {
              invSum += (D_inv[i] = 1-D[i]);
            }
            
            for(int i = 0; i < N; i++) {
              D_inv[i] /= invSum;
            }
            double p_d = 0, p_id = 0, n_d = 0, n_id = 0;
            
            for(int i = 0; i < N; i++)
            {
                final DataPointPair<Integer> dpp = dataPoints.get(i);
                
                H_cur[i] = (weak.classify(dpp.getDataPoint()).getProb(1)*2-1);
                final double outPut = Math.signum(H_cur[i]);
                final int c = dpp.getPair();
                if(c == 1)//positive example case
                {
                    p_d  += outPut * D[i];
                    p_id += outPut * D_inv[i];
                }
                else
                {
                    n_d  += outPut * D[i];
                    n_id += outPut * D_inv[i];
                }
                
            }
            
            final double alpha_m = p_d * (1 - p_id) - n_d * (1 - n_id); 
            
            if(Math.signum(alpha_m) != Math.signum(p_d-n_d) || Math.abs((p_d - n_d)) < 1e-6 || alpha_m <= 0) {
              return;
            }
            
            weightSum = 0;
            for(int i = 0; i < N; i++)
            {
                final DataPoint dp = dataPoints.get(i).getDataPoint();
                double w_i = dp.getWeight();
                final int y_i = dataPoints.get(i).getPair()*2-1;
                w_i *= Math.exp(-y_i*alpha_m*H_cur[i]);
                if(Double.isInfinite(w_i)) {
                  w_i = 1;//Let it grow back
                } else if(w_i <= 0) {
                  w_i = 1e-3/N;//Dont let it go quit to zero
                }
                weightSum += w_i;
                dp.setWeight(w_i);
            }
            
            for(int i = 0; i < N; i++)
            {
                final DataPoint dp = dataPoints.get(i).getDataPoint();
                final double w_i = dp.getWeight();
                dp.setWeight(Math.max(w_i/weightSum, 1e-10));//no zeros allowed
            }
            
            hypWeights.add(alpha_m);
            hypoths.add(weak);
        }
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public ModestAdaBoost clone()
    {
        return new ModestAdaBoost(this);
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(final String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }

}
