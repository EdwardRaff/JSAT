
package jsat.math.optimization;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import static java.lang.Math.*;
import java.util.concurrent.ExecutorService;
import jsat.math.FunctionVec;

/**
 * The Rosenbrock function is a function with at least one minima with the value zero. It is often used as a benchmark for optimization problems. <br>
 * The minima is the vector of all ones. Once N %gt; 3, more then one minima can occur. 
 * 
 * @author Edward Raff
 */
public class RosenbrockFunction implements Function
{

	private static final long serialVersionUID = -5573482950045304948L;

	@Override
    public double f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }

    @Override
    public double f(Vec x)
    {
        int N = x.length();
        double f = 0.0;
        for(int i = 1; i < N; i++)
        {
            double x_p = x.get(i-1);
            double xi = x.get(i);
            f += pow(1.0-x_p, 2)+100.0*pow(xi-x_p*x_p, 2);
        }
        
        return f;
    }
    
    /**
     * Returns the gradient of the Rosenbrock function
     * @return the gradient of the Rosenbrock function
     */
    public FunctionVec getDerivative()
    {
        return GRADIENT;
    }
    
    /**
     * The gradient of the Rosenbrock function
     */
    public static final FunctionVec GRADIENT = new FunctionVec()
    {
        @Override
        public Vec f(double... x)
        {
            return f(DenseVector.toDenseVec(x));
        }

        @Override
        public Vec f(Vec x)
        {
            Vec s = x.clone();
            f(x, s);
            return s;
        }

        @Override
        public Vec f(Vec x, Vec drv)
        {
            int N = x.length();

            if (drv == null)
                drv = x.clone();
            drv.zeroOut();

            drv.set(0, -400 * x.get(0) * (x.get(1) - pow(x.get(0), 2)) - 2 * (1 - x.get(0)));

            for (int i = 1; i < N - 1; i++)
            {
                double x_p = x.get(i - 1);
                double x_i = x.get(i);
                double x_n = x.get(i + 1);
                drv.set(i, 200 * (x_i - x_p * x_p) - 400 * x_i * (x_n - x_i * x_i) - 2 * (1 - x_i));
            }

            drv.set(N - 1, 200 * (x.get(N - 1) - pow(x.get(N - 2), 2)));

            return drv;
        }

        @Override
        public Vec f(Vec x, Vec s, ExecutorService ex)
        {
            return f(x, s);
        }
    };
}
