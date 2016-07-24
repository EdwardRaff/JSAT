
package jsat.datatransform;

import java.util.Iterator;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.*;

/**
 * Dense sparce transform alters the vectors that store the numerical values. 
 * Based on a threshold in (0, 1), vectors will be converted from dense to 
 * sparce, sparce to dense, or left alone. 
 * 
 * @author Edward Raff
 */
public class DenseSparceTransform implements DataTransform
{

    private static final long serialVersionUID = -1177913691660616290L;
    private double factor;

    /**
     * Creates a new Dense Sparce Transform. The <tt>factor</tt> gives the maximal 
     * percentage of values that may be non zero for a vector to be sparce. Any 
     * vector meeting the requirement will be converted to a sparce vector, and 
     * others made dense. If the factor is greater than or equal to 1, then all 
     * vectors will be made sparce. If less than or equal to 0, then all will 
     * be made dense. 
     * 
     * @param factor the fraction of the vectors values that may be non zero to qualify as sparce
     */
    public DenseSparceTransform(double factor)
    {
        this.factor = factor;
    }

    @Override
    public void fit(DataSet data)
    {
        //no - op, nothing we need to learn
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec orig = dp.getNumericalValues();

        if (orig instanceof SparseVector)
        {
            SparseVector sv = (SparseVector) orig;
            if (sv.nnz() / (double) sv.length() < factor)///Stay sparce
                return dp;

            DenseVector dv = new DenseVector(sv.length());
            Iterator<IndexValue> iter = sv.getNonZeroIterator();
            while (iter.hasNext())
            {
                IndexValue indexValue = iter.next();
                dv.set(indexValue.getIndex(), indexValue.getValue());
            }
            return new DataPoint(dv, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());

        }
        //Else, we are currently dense
        int nnz = 0;
        for(int i  = 0; i < orig.length(); i++)
            if(orig.get(i) != 0)
                nnz++;
        if(nnz / (double)orig.length() > factor)//Stay dense
            return dp;
        //Else, to sparce
        SparseVector sv = new SparseVector(orig.length(), nnz);//TODO create a constructor for this 
        for(int i  = 0; i < orig.length(); i++)
            if(orig.get(i) != 0)
                sv.set(i, orig.get(i));

        return new DataPoint(sv, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }

    @Override
    public DenseSparceTransform clone()
    {
        return new DenseSparceTransform(factor);
    }
}
