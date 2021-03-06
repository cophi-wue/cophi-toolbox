# Information for Contributors

## Development Environment

* [Code and Issues are at Github](http://github.com/cophi-wue/cophi-toolbox)
* [Integration Build](http://dev.digital-humanities.de/ci/blue/organizations/jenkins/cophi_toolbox/activity)

### Start hacking

```bash
git clone -b testing git@github.com:cophi-wue/cophi-toolbox
cd cophi-toolbox
mkvirtualenv cophi-toolbox  # if you use virtualenvwrapper
workon cophi-toolbox        # if you use virtualenvwrapper
pip install -r requirement-dev.txt
```

### Running the tests

Installing from `requirements-dev.txt` also installs the testing framework `pytest`, which is configured in `setup.cfg`. You can run the tests locally from the command line:

* `pytest` runs all unit tests (functions starting/ending with `test_` or `_test`, respectively) as well as all doctests.


## Releasing / Pushing to Master

The _testing_ branch is the integration branch for current developments. The _master_ branch should always contain the latest stable version. 

Pushing to master is protected, you can only push heads that have an "green" status from the integration build. To do so, do the following (from a clean working copy):

1. Prepare everything in `testing`. Don't forget to tune the version number.
2. Merge testing into master. Use `--no-ff` to make sure we get a merge commit: `git checkout master; git merge --no-ff testing`
3. if there are conflicts, resolve them and commit (to master)
4. now, fast-forward-merge master into testing: `git checkout testing; git merge master`. testing and master should now point to the same commit.
5. push testing. This will trigger the integration build, which will hopefully work out fine.
6. when the build has successfully finished, push master.

If something goes wrong, `git reset --hard master origin/master` and try again.

## Publishing on PyPi

Simply run:

```
python setup.py upload
```

and enter your credentials.