package jsat.classifiers.neuralnetwork.regularizers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This regularizer restricts the norm of each neuron's weights to be bounded by
 * a fixed constant, and rescaled when the norm is exceeded.
 * 
 * @author Edward Raff
 */
public class Max2NormRegularizer implements WeightRegularizer
{

	private static final long serialVersionUID = 1989826758516880355L;
	private double maxNorm;

    public Max2NormRegularizer(double maxNorm)
    {
        setMaxNorm(maxNorm);
    }

    /**
     * Sets the maximum allowed 2 norm for a single neuron's weights
     * @param maxNorm the maximum norm per neuron's weights
     */
    public void setMaxNorm(double maxNorm)
    {
        if(Double.isNaN(maxNorm) || Double.isInfinite(maxNorm) || maxNorm <= 0)
            throw new IllegalArgumentException("The maximum norm must be a positive constant, not " + maxNorm);
        this.maxNorm = maxNorm;
    }

    /**
     * 
     * @return the maximum allowed 2 norm for a single neuron's weights
     */
    public double getMaxNorm()
    {
        return maxNorm;
    }

    @Override
    public void applyRegularization(Matrix W, Vec b)
    {
        for (int i = 0; i < W.rows(); i++)
        {
            Vec W_li = W.getRowView(i);
            double norm = W_li.pNorm(2);
            if (norm >= maxNorm)
            {
                W_li.mutableMultiply(maxNorm / norm);
                double oldB_i = b.get(i);
                b.set(i, oldB_i * maxNorm / norm);
            }
        }
    }
    
    @Override
    public void applyRegularization(final Matrix W, final Vec b, ExecutorService ex)
    {
        List<Future<?>> futures = new ArrayList<Future<?>>(W.rows());
        for (int indx = 0; indx < W.rows(); indx++)
        {
            final int i = indx;
            futures.add(ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    Vec W_li = W.getRowView(i);
                    double norm = W_li.pNorm(2);
                    if (norm >= maxNorm)
                    {
                        W_li.mutableMultiply(maxNorm / norm);
                        double oldB_i = b.get(i);
                        b.set(i, oldB_i * maxNorm / norm);
                    }
                }
            }));
        }
        
        
        try
        {
            for (Future<?> future : futures)
                future.get();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(Max2NormRegularizer.class.getName()).log(Level.SEVERE, null, ex1);
        }
        catch (ExecutionException ex1)
        {
            Logger.getLogger(Max2NormRegularizer.class.getName()).log(Level.SEVERE, null, ex1);
        }
    }

    @Override
    public double applyRegularizationToRow(Vec w, double b)
    {
        double norm = w.pNorm(2);
        if (norm >= maxNorm)
        {
            w.mutableMultiply(maxNorm / norm);
            return b * maxNorm / norm;
        }
        return b;
    }

    @Override
    public Max2NormRegularizer clone()
    {
        return new Max2NormRegularizer(maxNorm);
    }

}
