
package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class TfIdf extends WordWeighting
{

    private double totalDocuments;
    private double documentWordCount;
    private List<Integer> df;

    public void setWeight(int totalDocuments, List<Integer> df)
    {
        this.totalDocuments = totalDocuments;
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
        documentWordCount = vec.sum();
        vec.applyIndexFunction(this);
    }
}
