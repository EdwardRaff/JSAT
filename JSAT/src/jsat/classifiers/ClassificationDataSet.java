
package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class ClassificationDataSet //extends DataSet
{
    /**
     * The number of numerical values each data point must have
     */
    protected int numNumerVals;
    protected CategoricalData[] categories;
    /**
     * The categories for the predicted value
     */
    protected CategoricalData predicting;
    /**
     * Contains the classification of the example data points in {@link #classifiedExamples}
     */
    protected List<Integer> classification;
    /**
     * Contains a list of data points that have already been classified according to {@link #predicting}
     */
    protected List<DataPoint> classifiedExamples;

    public ClassificationDataSet(int numerical, CategoricalData[] categories, CategoricalData predicting)
    {
        this.predicting = predicting;
        this.categories = categories;
        
        classification = new ArrayList<Integer>();
        classifiedExamples = new ArrayList<DataPoint>();
    }
    
    
    public void addDataPoint(Vec v, int[] classes, int classification)
    {
        if(v.length() != numNumerVals)
            throw new RuntimeException("Data point does not contain enough numerical data points");
        if(classes.length != categories.length)
            throw new RuntimeException("Data point does not contain enough categorical data points");
        
        for(int i = 0; i < classes.length; i++)
            if(!categories[i].isValidCategory(i))
                throw new RuntimeException("Categoriy value given is invalid");
        
        classifiedExamples.add(new DataPoint(v, classes, categories));
        this.classification.add(classification);
        
    }
    
}
