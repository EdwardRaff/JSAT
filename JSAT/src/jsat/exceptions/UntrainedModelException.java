
package jsat.exceptions;

/**
 * This exception is thrown when someone attempts to use a model that has not been trained or constructed. 
 * @author Edward Raff
 */
public class UntrainedModelException extends RuntimeException
{


	private static final long serialVersionUID = 3693546100471013277L;

	public UntrainedModelException(final String message, final Throwable cause)
    {
        super(message, cause);
    }

    public UntrainedModelException(final Throwable cause)
    {
        super(cause);
    }

    public UntrainedModelException(final String message)
    {
        super(message);
    }

    public UntrainedModelException()
    {
        super();
    }
    
}
