
package jsat;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import jsat.classifiers.DataPoint;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import static java.lang.Math.*;
/**
 *
 * @author Edward Raff
 */
public class MainRegression
{

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        int threads = Runtime.getRuntime().availableProcessors();
        ExecutorService threadPool = Executors.newFixedThreadPool(threads, new ThreadFactory() { 

            public Thread newThread(Runnable r)
            {
                Thread thrd = new Thread(r);
                thrd.setDaemon(true);
                return thrd;
            }
        });
        
        
        String path = "/Users/Edward Raff/Dropbox/ARFF DataSets/uci/numeric/";
        String sFile = path + "quake.arff";
        
        File f = new File(sFile);

        List<DataPoint> dataPoints = ARFFLoader.loadArffFile(f);
        RegressionDataSet rds = new RegressionDataSet(dataPoints, dataPoints.get(0).numNumericalValues()-1);
        
        int folds = 10;
        List<RegressionDataSet> lcds = rds.cvSet(folds);
        
        
        Regressor regressor;
        
        regressor = new MultipleLinearRegression();
//        regressor = new NearestNeighbour(3, false);
        
        long trainingTime = 0, classificationTime = 0;
        
        
        DenseVector rSqrdVals = new DenseVector(folds);
        
        for(int i = 0; i < lcds.size(); i++)
        {
            RegressionDataSet trainSet = RegressionDataSet.comineAllBut(lcds, i);
            RegressionDataSet testSet = lcds.get(i);
            
            double ssTot = 0, ssErr = 0, ssReg = 0;
            Vec trueVals = testSet.regressionValues();
            

            long startTrain = System.currentTimeMillis();
//            regressor.train(trainSet);
            regressor.train(trainSet, threadPool);
            trainingTime += (System.currentTimeMillis() - startTrain);
            
            for(int j = 0; j < testSet.getSampleSize(); j++)
            {
                double trueVal = trueVals.get(j);
                double predicted = regressor.regress(testSet.getDataPoint(j));
                double residual = predicted - trueVal;
                
                ssTot += pow(abs(trueVal-trueVals.mean()), 2);
                ssErr += pow(abs(residual), 2);   
                ssReg += pow(abs(predicted-trueVals.mean()), 2);
            }
            
            double rSqrd = 1-ssReg/ssTot;
            rSqrdVals.set(i, rSqrd);
        }
        System.out.println("R^2 Vals: " + rSqrdVals);
        System.out.println("R^2 Mean: " + rSqrdVals.mean());
    }
}
