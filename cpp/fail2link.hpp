#include "precomp.hpp"

#include "opencv2/videoio/registry.hpp"
#include "videoio_registry.hpp"

using namespace cv;

bool VideoWriter::open(const String& filename, int apiPreference, int fourcc, double fps,
                       const Size& frameSize, const std::vector<int>& params)
{
    CV_INSTRUMENT_REGION();

    if (isOpened())
    {
        release();
    }

    const VideoWriterParameters parameters(params);
    for (const auto& info : videoio_registry::getAvailableBackends_Writer())
    {
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            CV_WRITER_LOG_DEBUG(NULL,
                                cv::format("VIDEOIO(%s): trying writer with filename='%s' "
                                           "fourcc=0x%08x fps=%g sz=%dx%d isColor=%d...",
                                           info.name, filename.c_str(), (unsigned)fourcc, fps,
                                           frameSize.width, frameSize.height,
                                           parameters.get(VIDEOWRITER_PROP_IS_COLOR, true)));
            CV_Assert(!info.backendFactory.empty());
            const Ptr<IBackend> backend = info.backendFactory->getBackend();
            if (!backend.empty())
            {
                try
                {
                    iwriter = backend->createWriter(filename, fourcc, fps, frameSize, parameters);
                    if (!iwriter.empty())
                    {

                        CV_WRITER_LOG_DEBUG(NULL,
                                            cv::format("VIDEOIO(%s): created, isOpened=%d",
                                                       info.name, iwriter->isOpened()));
                        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG)
                        {
                            for (int key: parameters.getUnused())
                            {
                                CV_LOG_WARNING(NULL,
                                               cv::format("VIDEOIO(%s): parameter with key '%d' was unused",
                                                          info.name, key));
                            }
                        }
                        if (iwriter->isOpened())
                        {
                            return true;
                        }
                        iwriter.release();
                    }
                    else
                    {
                        CV_WRITER_LOG_DEBUG(NULL, cv::format("VIDEOIO(%s): can't create writer",
                                                             info.name));
                    }
                }
                catch (const cv::Exception& e)
                {
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n",
                                              info.name, e.what()));
                }
                catch (const std::exception& e)
                {
                    CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n",
                                                    info.name, e.what()));
                }
                catch (...)
                {
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n",
                                              info.name));
                }
            }
            else
            {
                CV_WRITER_LOG_DEBUG(NULL,
                                    cv::format("VIDEOIO(%s): backend is not available "
                                               "(plugin is missing, or can't be loaded due "
                                               "dependencies or it is not compatible)",
                                               info.name));
            }
        }
    }

    if (cv::videoio_registry::checkDeprecatedBackend(apiPreference))
    {
        CV_LOG_DEBUG(NULL,
            cv::format("VIDEOIO(%s): backend is removed from OpenCV",
                cv::videoio_registry::getBackendName((VideoCaptureAPIs) apiPreference).c_str()));
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO: choosen backend does not work or wrong."
            "Please make sure that your computer support chosen backend and OpenCV built "
            "with right flags.");
    }

    return false;
}
