## MNP Batch Template for Creating Docker Images

Template scripts to setup Docker Images compatible with running on MNP Batch

## License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.

## Deployment

````bash
git clone https://github.com/aws-samples/aws-mnpbatch-template.git
cd aws-mnpbatch-template
cp examples/mnp-tensorflow/Docker .
docker build -t nvidia/mnp-batch-tensorflow .
```
