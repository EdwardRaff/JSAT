
package jsat.classifiers.calibration;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;

/**
 * Isotonic Calibration is non-parametric, and only assumes that the underlying 
 * distribution from negative to positive examples is strictly a non-decreasing 
 * function. It will then attempt to model the distribution. This may over-fit 
 * for small data sizes, and imposes an additional <i>O(log n)</i> search look 
 * up when performing classification, where n is &lt;= the number of data points
 * in the data set. 
 * <br><br>
 * Isotonic Calibration inherently creates non-adjacent bins of varying size.
 * Smooth transitions in output probability are created by simple linear 
 * interpolation between bin values. 
 * <br><br>
 * See: Niculescu-Mizil, A.,&amp;Caruana, R. (2005). <i>Predicting Good 
 * Probabilities with Supervised Learning</i>. International Conference on 
 * Machine Learning (pp. 625–632). Retrieved from 
 * <a href="http://dl.acm.org/citation.cfm?id=1102430">here</a>
 * 
 * @author Edward Raff
 */
public class IsotonicCalibration extends BinaryCalibration
{

	private static final long serialVersionUID = -1295979238755262335L;
	private double[] outputs;
    private double[] scores;

    /**
     * Creates a new Isotonic Calibration object
     * @param base the base model to calibrate the outputs of
     * @param mode the calibration mode to use
     */
    public IsotonicCalibration(BinaryScoreClassifier base, CalibrationMode mode)
    {
        super(base, mode);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected IsotonicCalibration(IsotonicCalibration toCopy)
    {
        super(toCopy.base.clone(), toCopy.mode);
        if (toCopy.outputs != null)
            this.outputs = Arrays.copyOf(toCopy.outputs, toCopy.outputs.length);
        if (toCopy.scores != null)
            this.scores = Arrays.copyOf(toCopy.scores, toCopy.scores.length);
    }
    
    private static class Point implements Comparable<Point>
    {
        public double weight;
        public double score;
        public double output;
        
        public double min, max;
        

        public Point(double score, double output)
        {
            this.weight = 1;
            min = max = this.score = score;
            this.output = output;
        }
        
        public void merge(Point next)
        {
            double newWeight = this.weight +next.weight;
            this.score = (this.weight*this.score + next.weight*next.score)/newWeight;
            this.output = (this.weight*this.output + next.weight*next.output)/newWeight;
            this.weight = newWeight;
            this.min = Math.min(this.min, next.min);
            this.max = Math.max(this.max, next.max);
        }
        
        public boolean nextViolates(Point next)
        {
            return this.output >= next.output;
        }

        @Override
        public int compareTo(Point o)
        {
            return Double.compare(score, o.score);
        }

    }

    @Override
    protected void calibrate(boolean[] label, double[] deci, int len)
    {
        List<Point> points = new ArrayList<Point>(len);
        for(int i = 0; i < len; i++)
            points.add(new Point(deci[i], label[i] ? 1 : 0));
        Collections.sort(points);
        boolean violators = true;
        while(violators)
        {
            violators = false;
            for(int i = 0; i < points.size()-1; i++)
            {
                if(points.get(i).nextViolates(points.get(i+1)))
                {
                    violators = true;
                    points.get(i).merge(points.remove(i+1));
                    i--;
                }
            }
        }
        
        scores = new double[points.size()*2];
        outputs = new double[points.size()*2];
        
        int pos = 0;
        for(Point p : points)
        {
            scores[pos] = p.min;
            outputs[pos++] = p.output;
            scores[pos] = p.max;
            outputs[pos++] = p.output;
        }
        
    }

    @Override
    public IsotonicCalibration clone()
    {
        return new IsotonicCalibration(this);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        double score = base.getScore(data);
        
        CategoricalResults cr = new CategoricalResults(2);
        int indx = Arrays.binarySearch(scores, score);
        if(indx < 0)
            indx = (-(indx) - 1);
        
        if(indx == scores.length)
        {
            double maxScore = scores[scores.length-1];
            if(score > maxScore*3)
                cr.setProb(1, 1.0);
            else
            {
                double p = (maxScore*3-score)/(maxScore*2)*outputs[scores.length-1];
                cr.setProb(0, 1-p);
                cr.setProb(1, p);
            }
        }
        else if(indx == 0)
        {
            double minScore = scores[0];
            if(score < minScore/3)
                cr.setProb(0, 1.0);
            else
            {
                double p = (minScore-score)/(minScore-minScore/3)*outputs[0];
                cr.setProb(0, 1-p);
                cr.setProb(1, p);
            }
        }
        else
        {
            double score0 = scores[indx-1];
            double score1 = scores[indx];
            
            if(score0 == score1)
            {
                cr.setProb(0, 1-outputs[indx]);
                cr.setProb(1, outputs[indx]);
                return cr;
            }
            
            double weight = (score1-score)/(score1-score0);
            double p = outputs[indx-1]*weight + outputs[indx]*(1-weight);
            cr.setProb(0, 1-p);
            cr.setProb(1, p);   
        }
        
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return base.supportsWeightedData();
    }

}
