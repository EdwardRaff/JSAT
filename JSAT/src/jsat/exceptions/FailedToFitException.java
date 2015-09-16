package jsat.exceptions;

/**
 *
 * @author Edward Raff
 */
public class FailedToFitException extends RuntimeException {

  private static final long serialVersionUID = 2982189541225068993L;
  private Exception faultException;

  public FailedToFitException(final Exception faultException) {
    super(faultException.getMessage());
    this.faultException = faultException;
  }

  public FailedToFitException(final Exception faultException, final String message) {
    super(message);
    this.faultException = faultException;
  }

  public FailedToFitException(final Exception faultException, final String message, final Throwable cause) {
    super(message, cause);
    this.faultException = faultException;
  }

  public FailedToFitException(final Exception faultException, final Throwable cause) {
    super(cause);
    this.faultException = faultException;
  }

  public FailedToFitException(final String string) {
    super(string);
  }

  /**
   * Returns the exception that caused the issue. If no exception occurred that
   * caused the failure to fit, the value returned will be null.
   *
   * @return the exception that caused the issue.
   */
  public Exception getFaultException() {
    return faultException;
  }

}
