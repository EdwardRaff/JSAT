package jsat.classifiers.neuralnetwork.initializers;

import java.util.Random;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This initializer samples the weights from an adjusted uniform distribution 
 * in order to provided better behavior of neuron activation and gradients<br>
 * <br>
 * See: Glorot, X., &amp; Bengio, Y. (2010). <i>Understanding the difficulty of 
 * training deep feedforward neural networks</i>. Journal of Machine Learning 
 * Research - Proceedings Track, 9, 249–256. Retrieved from 
 * <a href="http://jmlr.csail.mit.edu/proceedings/papers/v9/glorot10a/glorot10a.pdf">
 * here</a>
 * @author Edward Raff
 */
public class TanhInitializer implements WeightInitializer, BiastInitializer
{


	private static final long serialVersionUID = -4770682311082616208L;

	@Override
    public void init(Matrix w, Random rand)
    {
        double cnt = Math.sqrt(6)/Math.sqrt(w.rows()+w.cols());
        for(int i = 0; i < w.rows(); i++)
            for(int j = 0; j < w.cols(); j++)
                w.set(i, j, rand.nextDouble()*cnt*2-cnt);
        
    }

    @Override
    public void init(Vec b, int fanIn, Random rand)
    {
        double cnt = Math.sqrt(6)/Math.sqrt(b.length()+fanIn);
        for(int i = 0; i < b.length(); i++)
            b.set(i, rand.nextDouble()*cnt*2-cnt);
    }

    @Override
    public TanhInitializer clone()
    {
        return new TanhInitializer();
    }
    
}
