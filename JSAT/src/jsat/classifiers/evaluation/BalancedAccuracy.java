package jsat.classifiers.evaluation;

import java.util.Arrays;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * This class implements the Balanced Accuracy metric. If the number of test
 * points has an equal total weight for each class, then Balanced Accuracy
 * returns the same result as {@link Accuracy}. If not, this class will re-scale
 * the importance of errors in each class and return an accuracy score as if
 * each class had equal total weight. That is to say, it evaluates the data as
 * if it was balanced. <br>
 * <br>
 * See: Brodersen, K. H., Ong, C. S., Stephan, K. E., &
 * Buhmann, J. M. (2010).
 * <i>The Balanced Accuracy and Its Posterior Distribution</i>. In Proceedings
 * of the 2010 20th International Conference on Pattern Recognition (pp.
 * 3121–3124). Washington, DC, USA: IEEE Computer Society.
 * <a href="http://doi.org/10.1109/ICPR.2010.764">http://doi.org/10.1109/ICPR.2010.764</a>
 * @author Edward Raff
 */
public class BalancedAccuracy implements ClassificationScore
{
    private int classes;
    double[] class_correct;
    double[] total_class_weight;
    

    public BalancedAccuracy()
    {
        
    }
    
    
    public BalancedAccuracy(BalancedAccuracy toClone)
    {
        this.classes = toClone.classes;
        if(toClone.class_correct != null)
            this.class_correct = Arrays.copyOf(toClone.class_correct, toClone.class_correct.length);
        if(toClone.total_class_weight != null)
            this.total_class_weight = Arrays.copyOf(toClone.total_class_weight, toClone.total_class_weight.length);
    }
    

    @Override
    public double getScore()
    {
        double score = 0;
        for(int i = 0; i < classes; i++)
        {
            if(total_class_weight[i] > 1e-15)
                score += class_correct[i]/total_class_weight[i];
            else
                score += 1;
        }
        score /= classes;
        return score;
    }

    @Override
    public boolean lowerIsBetter()
    {
        return false;
    }
    
    @Override
    public BalancedAccuracy clone()
    {
        return new BalancedAccuracy(this);
    }

    @Override
    public String getName()
    {
        return "BalancedAccuracy";
    }

    @Override
    public int hashCode()
    {
        return getName().hashCode();
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return true;
        }
        return false;
    }

    @Override
    public void prepare(CategoricalData toPredict)
    {
        classes = toPredict.getNumOfCategories();
        total_class_weight = new double[classes];
        class_correct = new double[classes];
    }

    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        total_class_weight[trueLabel] += weight;
        if(prediction.mostLikely() == trueLabel)
            class_correct[trueLabel] += weight;
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        if(other instanceof BalancedAccuracy)
        {
            BalancedAccuracy o = (BalancedAccuracy) other;
            for(int i = 0; i < classes; i++)
            {
                this.class_correct[i] += o.class_correct[i];
                this.total_class_weight[i] += o.total_class_weight[i];
            }
        }
        
    }
}
