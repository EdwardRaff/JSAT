
package jsat.text.wordweighting;

import static java.lang.Math.log;
import java.util.List;
import jsat.linear.Vec;

/**
 * Applies Term Frequency Inverse Document Frequency (TF IDF) weighting to the 
 * word vectors. 
 * 
 * @author Edward Raff
 */
public class TfIdf extends WordWeighting
{

    private double totalDocuments;
    private List<Integer> df;

    @Override
    public void setWeight(List<? extends Vec> allDocuments, List<Integer> df)
    {
        this.totalDocuments = allDocuments.size();
        this.df = df;
    }

    @Override
    public double indexFunc(double value, int index)
    {
        if (index < 0)
            return 0.0;

        double tf = 1+log(value);
        double idf = log(totalDocuments / df.get(index));

        return tf * idf;
    }

    @Override
    public void applyTo(Vec vec)
    {
        vec.applyIndexFunction(this);
    }
}
