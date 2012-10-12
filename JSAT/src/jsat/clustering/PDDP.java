package jsat.clustering;

import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class PDDP extends ClustererBase
{

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        Vec mc = MatrixStatistics.meanVector(dataSet);
        
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
