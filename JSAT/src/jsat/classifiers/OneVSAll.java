
package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.svm.PlatSMO;
import jsat.distributions.kernels.LinearKernel;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class OneVSAll implements Classifier
{
    Classifier[] oneVsAlls;
    Classifier baseClassifier;
    CategoricalData predicting;
    
    public OneVSAll(Classifier baseClassifier)
    {
        this.baseClassifier = baseClassifier;
    }
    
    
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        double sum = 0;
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            if(oneVsAlls[i].classify(data).getProb(0) > 0)
            {
                double tmp = oneVsAlls[i].classify(data).getProb(0);
                cr.setProb(i, tmp);
                sum += tmp;
//                return cr;
            }
        
        cr.divideConst(sum);
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        oneVsAlls = new Classifier[dataSet.getPredicting().getNumOfCategories()];
        
        predicting = dataSet.getPredicting();
        
        List<List<DataPoint>> categorized = new ArrayList<List<DataPoint>>();
        for(int i = 0; i < oneVsAlls.length; i++)
        {
            List<DataPoint> tmp = dataSet.getSamples(i);
            ArrayList<DataPoint> oneCat = new ArrayList<DataPoint>(tmp.size());
            oneCat.addAll(tmp);
            categorized.add(oneCat);
        }
        
        int numer = dataSet.getDataPoint(0).getNumericalValues().length();
        CategoricalData[] categories = dataSet.getDataPoint(0).getCategoricalData();
        for(int i = 0; i < oneVsAlls.length; i++)
        {
            ClassificationDataSet cds = 
                    new ClassificationDataSet(numer, categories, new CategoricalData(2));
            for(DataPoint dp : categorized.get(i))//add the ones
                cds.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 0);
            //Add all the 'others'
            for(int j = 0; j < categorized.size(); j++)
                if(j != i)
                    for(DataPoint dp: categorized.get(j))
                        cds.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 1);

            baseClassifier.trainC(cds, threadPool);
//            PlatSMO cls = new PlatSMO(new LinearKernel());
            oneVsAlls[i] = baseClassifier.copy();
//            cls.trainC(cds);
//            oneVsAlls[i] = cls;
        }
        
        
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public Classifier copy()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public boolean supportsWeightedData()
    {
        return baseClassifier.supportsWeightedData();
    }
    
}
