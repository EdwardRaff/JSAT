package jsat.classifiers.svm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IndexTable;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class SBP extends SupportVectorLearner implements Classifier, Parameterized
{
    double v = 0.1;
    int iterations;
    int T_0;

    public SBP(KernelTrick kernel, CacheMode cacheMode)
    {
        super(kernel, cacheMode);
    }

    public SBP(SBP other)
    {
        this(other.getKernel().clone(), other.getCacheMode());
        if(other.alphas != null)
            this.alphas = Arrays.copyOf(other.alphas, other.alphas.length);
        this.iterations = other.iterations;
        this.T_0 = other.T_0;
    }
    
    

    @Override
    public Classifier clone()
    {
        return new SBP(this);
    }

    public void setIterations(int iterations)
    {
        this.iterations = iterations;
    }

    public int getIterations()
    {
        return iterations;
    }

    public void setV(double v)
    {
        this.v = v;
    }

    public double getV()
    {
        return v;
    }
    

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");
        
        double sum = 0;
        CategoricalResults cr = new CategoricalResults(2);
        
        for (int i = 0; i < vecs.length; i++)
            sum += alphas[i] * kEval(vecs[i], data.getNumericalValues());


        //SVM only says yess / no, can not give a percentage
        if(sum < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        
        return cr;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        final int n = dataSet.getSampleSize();
        /*
         * Respone values
         */
        double[] C = new double[n];
        double[] CSum = new double[n];
        alphas = new double[n];
        double[] alphasSum = new double[n];

        double[] y = new double[n];
        vecs = new Vec[n];
        for(int i = 0; i < n; i++)
        {
            y[i] = dataSet.getDataPointCategory(i)*2-1;
            vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
        }
        
        Random rand = new XORWOW();
        double maxKii = 0;
        for(int i = 0; i < n; i++)
            maxKii = Math.max(maxKii, kEval(vecs[i], vecs[i]));//avoid starting the cache on the diagonal
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        final double eta_0 = 1/Math.sqrt(maxKii);
        
        double rSqrd = 0;
        
        for(int t = 1; t <= iterations; t++)
        {
            final double eta = eta_0/Math.sqrt(t);
            final double gamma = findGamma(C, n*v);

            //Samply uniformly from C[i] <= gamma
            int attempts = 0;//you get 5 attempts to find one quickly
            int i;
            do
            {
                i = rand.nextInt(n);
                attempts++;
            }
            while(C[i] > gamma && attempts < 5);
            
            if(C[i] > gamma)//find one the slow way
            {
                int candidates = 0;
                for(int j = 0; j < C.length; j++)
                    if(C[i] <= gamma)
                        candidates++;
                int randCand = rand.nextInt(candidates);
                i = 0;
                for(int j = 0; j < C.length && i < randCand; j++)
                    if(C[i] <= gamma)
                        i++;
            }
            
            
            alphas[i] += eta;
            rSqrd += 2*eta*C[i]+eta*eta*kEval(i, i);
            final double y_i = y[i];
            for(int j = 0; j < n; j++)
                C[j] += eta*y_i*y[j]*kEval(i, j);
            
            if(rSqrd > 1)//1^2 = 1, so jsut use sqrd version
            {
                final double rInv = 1/Math.sqrt(rSqrd);
                
                for(int j = 0; j < n; j++)
                {
                    C[j] *= rInv;
                    alphas[j] *= rInv;
                }
                
                rSqrd = 1;
            }

            
            for(int j = 0; j < n; j++)
            {
                alphasSum[j] += alphas[j];
                CSum[j] += C[j];
            }
        }
        
        //Take the averages
        for (int j = 0; j < n; j++)
        {
            alphas[j] = alphasSum[j]/iterations;
            C[j] = CSum[j]/iterations;
        }
        double gamma = findGamma(C, n*v);
        for (int j = 0; j < n; j++)
            alphas[j] /= gamma*y[j];
        
        //Clean up to only the SVs
        int supportVectorCount = 0;
        for(int i = 0; i < vecs.length; i++)
            if(alphas[i] != 0)//its a support vector
            {
                vecs[supportVectorCount] = vecs[i];
                alphas[supportVectorCount++] = alphas[i];
            }

        vecs = Arrays.copyOfRange(vecs, 0, supportVectorCount);
        alphas = Arrays.copyOfRange(alphas, 0, supportVectorCount);
        System.out.println(supportVectorCount + "/" + n + "  SVs");
        it = null;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
    private IndexTable it;

    private double findGamma(double[] C, double d)
    {
        if(it == null )
            it = new IndexTable(C);
        else
        {
            it.reset();
            it.sort(C);
        }
        
        double sum = 0;
        double prev = 0;
        
        for(int i = 0; i < it.length(); i++)
        {
            double max = C[it.index(i)];
            sum += max;
            
            double score = max*i-sum;
            
            if(score >= d)
                return prev;
            else
                prev = score;
        }
        
        return prev;
    }
    
    private List<Parameter> params = Parameter.getParamsFromMethods(this);
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);
    
    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> retParams = new ArrayList<Parameter>(params);
        retParams.addAll(getKernel().getParameters());
        return retParams;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        Parameter toRet = paramMap.get(paramName);
        if(toRet == null)
            toRet = getKernel().getParameter(paramName);
        return toRet;
    }
}
