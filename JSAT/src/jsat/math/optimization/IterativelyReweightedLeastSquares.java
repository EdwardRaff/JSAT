
package jsat.math.optimization;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.LUPDecomposition;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Provides an implementation of the Iteratively Reweighted Least Squares (IRLS) algorithm for solving certain classes 
 * of optimization problems. See <a href="http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares">Wikepida</a> 
 * for more information. 
 * 
 * @author Edward Raff
 */
public class IterativelyReweightedLeastSquares implements Optimizer
{
    

	private static final long serialVersionUID = -6872953184371630318L;
	/**
     * The hessian matrix
     */
    private DenseMatrix hessian;
    /**
     * Contains the values of the coefficients for each data point
     */
    private DenseMatrix coefficentMatrix;
    private DenseVector derivatives;
    private DenseVector errors;
    private DenseVector gradiant;

    public IterativelyReweightedLeastSquares()
    {
        
    }
    
    public Vec optimize(double eps, int iterationLimit, Function f, Function fd, Vec vars, List<Vec> inputs, Vec outputs)
    {
        return optimize(eps, iterationLimit, f, fd, vars, inputs, outputs, null);
    }
    
    public Vec optimize(double eps, int iterationLimit, Function f, Function fd, Vec vars, List<Vec> inputs, Vec outputs, ExecutorService threadpool)
    {
        //TODO make it actually use the threadpool!
        hessian = new DenseMatrix(vars.length(), vars.length());
        coefficentMatrix = new DenseMatrix(inputs.size(), vars.length());
        for(int i = 0; i < inputs.size(); i++)
        {
            Vec x_i = inputs.get(i);
            coefficentMatrix.set(i, 0, 1.0);
            for(int j = 1; j < vars.length(); j++)
                coefficentMatrix.set(i, j, x_i.get(j-1));
        }
        
        derivatives = new DenseVector(inputs.size());
        errors = new DenseVector(outputs.length());
        gradiant = new DenseVector(vars.length());
        
        double maxChange = Double.MAX_VALUE;
        //No reason to do the if check in a tightish loop 
        if (threadpool != null && !(threadpool instanceof FakeExecutor))//Serial 
        {
            do
            {
                maxChange = iterationStep(f, fd, vars, inputs, outputs, threadpool);
            }
            while (!Double.isNaN(maxChange) && maxChange > eps && iterationLimit-- > 0);
        }
        else//Parallel
        {
            do
            {
                maxChange = iterationStep(f, fd, vars, inputs, outputs);
            }
            while (!Double.isNaN(maxChange) && maxChange > eps && iterationLimit-- > 0);
        }

        return vars;
    }
    
    private double iterationStep(Function f,  Function fd, Vec vars, List<Vec> inputs, Vec outputs)
    {
        Vec delta = null;
        for(int i = 0; i < inputs.size(); i++)
        {
            Vec x_i = inputs.get(i);
            double y = f.f(x_i);
            double error = y - outputs.get(i);
            errors.set(i, error);
            
            derivatives.set(i, fd.f(x_i));
        }
        
        
        for (int j = 0; j < hessian.rows(); j++)
        {
            double gradTmp = 0;
            for (int k = 0; k < coefficentMatrix.rows(); k++)
            {
                double coefficient_kj = coefficentMatrix.get(k, j);
                gradTmp+= coefficient_kj*errors.get(k);
                
                double multFactor = derivatives.get(k) * coefficient_kj;
                
                for (int i = 0; i < hessian.rows(); i++)
                    hessian.increment(j, i, coefficentMatrix.get(k, i) * multFactor);
            }
            
            gradiant.set(j, gradTmp);
        }
        
        LUPDecomposition lupDecomp = new LUPDecomposition(hessian.clone());//We sent a clone of the hessian b/c we make incremental updates every iteration
        if(Math.abs(lupDecomp.det()) < 1e-14 )
        {
            //TODO use a pesudo inverse instead of giving up
            return Double.NaN;//Indicate that we need to stop
        }
        else//nomral case, solve!
        {
            delta = lupDecomp.solve(gradiant);
        }
        
        vars.mutableSubtract(delta);
        
        return Math.max(delta.max(), Math.abs(delta.min()));
    }
    
    
    private double iterationStep(Function f,  Function fd, Vec vars, List<Vec> inputs, Vec outputs, ExecutorService threadpool)
    {
        Vec delta = null;
        for(int i = 0; i < inputs.size(); i++)
        {
            Vec x_i = inputs.get(i);
            double y = f.f(x_i);
            double error = y - outputs.get(i);
            errors.set(i, error);
            
            derivatives.set(i, fd.f(x_i));
        }
        int overFlow = hessian.rows()%SystemInfo.LogicalCores;
        int size = hessian.rows()/SystemInfo.LogicalCores;
        int start  = 0;
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        for(int t = 0; t < SystemInfo.LogicalCores; t++)
        {
            final int START = start;
            final int TO = (overFlow-- > 0 ? 1 : 0) + START + size;
            start = TO;
            threadpool.submit(new Runnable() {

                public void run()
                {
                    for (int j = START; j < TO; j++)
                    {
                        double gradTmp = 0;
                        for (int k = 0; k < coefficentMatrix.rows(); k++)
                        {
                            double coefficient_kj = coefficentMatrix.get(k, j);
                            gradTmp += coefficient_kj * errors.get(k);

                            double multFactor = derivatives.get(k) * coefficient_kj;

                            for (int i = 0; i < hessian.rows(); i++)
                                hessian.increment(j, i, coefficentMatrix.get(k, i) * multFactor);
                        }

                        gradiant.set(j, gradTmp);
                    }
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            ex.printStackTrace();
        }

        LUPDecomposition lupDecomp = new LUPDecomposition(hessian.clone(), threadpool);//We sent a clone of the hessian b/c we make incremental updates every iteration
        if (Math.abs(lupDecomp.det()) < 1e-14)
        {
            //TODO use a pesudo inverse instead of giving up
            return Double.NaN;//Indicate that we need to stop
        }
        else//nomral case, solve!
        {
            delta = lupDecomp.solve(gradiant);
        }

        vars.mutableSubtract(delta);

        return Math.max(delta.max(), Math.abs(delta.min()));
    }
}
